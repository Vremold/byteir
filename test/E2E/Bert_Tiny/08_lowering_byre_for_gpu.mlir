// RUN: byteir-opt %s -func-tag="attach-attr=__placeholder__byre.entry_point func-name=main" -gen-ptx-config -convert-to-byre -cse | FileCheck %s

// CHECK-LABEL: func @main

module attributes {gpu.container_module}  {
  func private @MatmulOp0(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<128x30522xf32>
    %1 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %1 : memref<30522x128xf32>
  }
  func private @Unknown0(%arg0: memref<2x128xi64>, %arg1: memref<256xi64>, %arg2: memref<256xi64>, %arg3: memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %c8 = constant 8 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %0 = memref.alloc() : memref<2x128xui32>
    %1 = memref.alloc() : memref<2x128xi1>
    %2 = memref.collapse_shape %1 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %3 = memref.expand_shape %arg3 [[0, 1]] : memref<256xf64> into memref<2x128xf64>
    %4 = memref.alloc() : memref<2x128xi64>
    %5 = memref.collapse_shape %4 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %6 = memref.expand_shape %5 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %7 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %8 = memref.expand_shape %arg1 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %9 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    gpu.launch_func  @Unknown0_kernel::@Unknown0_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128xi64>, %0 : memref<2x128xui32>, %8 : memref<2x128xi64>, %7 : memref<2x128xi64>, %4 : memref<2x128xi64>, %3 : memref<2x128xf64>, %1 : memref<2x128xi1>)
    return %9, %6, %2 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
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
  func private @Unknown1(%arg0: memref<128xi64>, %arg1: memref<256xi64>, %arg2: memref<256xi64>, %arg3: memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %c8 = constant 8 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %0 = memref.alloc() : memref<2x128xui32>
    %1 = memref.alloc() : memref<2x128xi1>
    %2 = memref.collapse_shape %1 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %3 = memref.expand_shape %arg3 [[0, 1]] : memref<256xf64> into memref<2x128xf64>
    %4 = memref.alloc() : memref<2x128xi64>
    %5 = memref.collapse_shape %4 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %6 = memref.expand_shape %5 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %7 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %8 = memref.expand_shape %arg1 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %9 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    gpu.launch_func  @Unknown1_kernel::@Unknown1_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xi64>, %0 : memref<2x128xui32>, %8 : memref<2x128xi64>, %7 : memref<2x128xi64>, %4 : memref<2x128xi64>, %3 : memref<2x128xf64>, %1 : memref<2x128xi1>)
    return %9, %6, %2 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
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
  func private @Unknown2(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %c1024 = constant 1024 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
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
  func private @Unknown3(%arg0: memref<1x128xi64>, %arg1: memref<128xi64>, %arg2: memref<128xi64>, %arg3: memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {byre_elementwise_fusion} {
    %c4 = constant 4 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    %1 = memref.alloc() : memref<128xi1>
    %2 = memref.alloc() : memref<128xi64>
    %3 = memref.expand_shape %2 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    %4 = memref.alloc() : memref<128xui32>
    gpu.launch_func  @Unknown3_kernel::@Unknown3_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%0 : memref<128xi64>, %4 : memref<128xui32>, %arg1 : memref<128xi64>, %arg2 : memref<128xi64>, %2 : memref<128xi64>, %arg3 : memref<128xf64>, %1 : memref<128xi1>)
    return %4, %3, %1 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
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
  func private @Unknown4(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> memref<2x128x30522xf32> attributes {byre_elementwise_fusion} {
    %c244176 = constant 244176 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
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
      %c0 = constant 0 : index
      %c7813632 = constant 7813632 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c7813632 : index
      scf.if %6 {
        %c30522 = constant 30522 : index
        %7 = remi_signed %5, %c30522 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c30522 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c30522 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %c128 = constant 128 : index
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
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x30522xf32>
        %28 = memref.load %arg1[%10] : memref<30522xf32>
        %29 = addf %27, %28 : f32
        memref.store %29, %arg2[%26, %20, %10] : memref<2x128x30522xf32>
      }
      gpu.return
    }
  }
  func private @Unknown5(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %c1024 = constant 1024 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
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
  func private @Unknown6(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %c1024 = constant 1024 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
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
  func private @Unknown7(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256x128xf32>, %arg3: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {byre_elementwise_fusion} {
    %c1024 = constant 1024 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %0 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %1 = memref.alloc() : memref<2x128x128xf32>
    %2 = memref.collapse_shape %1 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %3 = memref.expand_shape %arg3 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %4 = memref.alloc() : memref<2x128x128xf32>
    %5 = memref.collapse_shape %4 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %6 = memref.expand_shape %arg2 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    gpu.launch_func  @Unknown7_kernel::@Unknown7_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%0 : memref<2x128xi1>, %arg1 : memref<2x128x128xf32>, %6 : memref<2x128x128xf32>, %4 : memref<2x128x128xf32>, %3 : memref<2x128xi1>, %1 : memref<2x128x128xf32>)
    return %5, %2 : memref<256x128xf32>, memref<256x128xf32>
  }
  gpu.module @Unknown7_kernel {
    gpu.func @Unknown7_kernel(%arg0: memref<2x128xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128xi1>, %arg5: memref<2x128x128xf32>) kernel {
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
  func private @Unknown8(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) -> memref<128x128xf32> attributes {byre_elementwise_fusion} {
    %c512 = constant 512 : index
    %c1 = constant 1 : index
    %c32 = constant 32 : index
    %0 = memref.alloc() : memref<128x128xf32>
    gpu.launch_func  @Unknown8_kernel::@Unknown8_kernel blocks in (%c512, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xi1>, %arg1 : memref<128x128xf32>, %arg2 : memref<128x128xf32>, %0 : memref<128x128xf32>)
    return %0 : memref<128x128xf32>
  }
  gpu.module @Unknown8_kernel {
    gpu.func @Unknown8_kernel(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>, %arg3: memref<128x128xf32>) kernel {
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
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<1x512xi64>, %arg2: memref<1x512xi64>, %arg3: memref<30522x128xf32>, %arg4: memref<2x128xf32>, %arg5: memref<512x128xf32>, %arg6: memref<128xf32>, %arg7: memref<128xf32>, %arg8: memref<128x128xf32>, %arg9: memref<128xf32>, %arg10: memref<128x128xf32>, %arg11: memref<128xf32>, %arg12: memref<128x128xf32>, %arg13: memref<128xf32>, %arg14: memref<2x1x1x128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<2x1x1x128xf32>, %arg32: memref<128x128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<512x128xf32>, %arg37: memref<512xf32>, %arg38: memref<128x512xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<30522xf32>, %arg47: memref<2x128x30522xf32>) -> (memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<30522x128xf32>
    %1 = memref.alloc() : memref<30522xf32>
    %2 = memref.alloc() : memref<512x128xf32>
    %3 = memref.alloc() : memref<128x128xf32>
    %4 = memref.alloc() : memref<2x128xf32>
    %5 = memref.alloc() : memref<30522x128xf32>
    %6 = memref.alloc() : memref<30522x128xf32>
    %7 = memref.alloc() : memref<2x128x128xf32>
    %8 = memref.alloc() : memref<128xf32>
    %9 = memref.alloc() : memref<128xf32>
    %10 = memref.alloc() : memref<2x128x128xf32>
    %11 = memref.alloc() : memref<128xf32>
    %12 = memref.alloc() : memref<128x128xf32>
    %13 = memref.alloc() : memref<2x128x128xf32>
    %14 = memref.alloc() : memref<128xf32>
    %15 = memref.alloc() : memref<128x128xf32>
    %16 = memref.alloc() : memref<2x128x128xf32>
    %17 = memref.alloc() : memref<128xf32>
    %18 = memref.alloc() : memref<128x128xf32>
    %19 = memref.alloc() : memref<2x128x128xf32>
    %20 = memref.alloc() : memref<2x2x128x64xf32>
    %21 = memref.alloc() : memref<2x2x128x64xf32>
    %22 = memref.alloc() : memref<2x2x128x128xf32>
    %23 = memref.alloc() : memref<2x2x128x64xf32>
    %24 = memref.alloc() : memref<2x2x128x128xf32>
    %25 = memref.alloc() : memref<2x2x128x64xf32>
    %26 = memref.alloc() : memref<2x128x2x64xf32>
    %27 = memref.alloc() : memref<128xf32>
    %28 = memref.alloc() : memref<128x128xf32>
    %29 = memref.alloc() : memref<2x128x128xf32>
    %30 = memref.alloc() : memref<2x128x128xf32>
    %31 = memref.alloc() : memref<128xf32>
    %32 = memref.alloc() : memref<128xf32>
    %33 = memref.alloc() : memref<2x128x128xf32>
    %34 = memref.alloc() : memref<2x128x128xf32>
    %35 = memref.alloc() : memref<512xf32>
    %36 = memref.alloc() : memref<512x128xf32>
    %37 = memref.alloc() : memref<2x128x128xf32>
    %38 = memref.alloc() : memref<128xf32>
    %39 = memref.alloc() : memref<128x512xf32>
    %40 = memref.alloc() : memref<2x128x512xf32>
    %41 = memref.alloc() : memref<2x128x128xf32>
    %42 = memref.alloc() : memref<128xf32>
    %43 = memref.alloc() : memref<128xf32>
    %44 = memref.alloc() : memref<2x128x128xf32>
    %45 = memref.alloc() : memref<128xf32>
    %46 = memref.alloc() : memref<128x128xf32>
    %47 = memref.alloc() : memref<2x128x128xf32>
    %48 = memref.alloc() : memref<128xf32>
    %49 = memref.alloc() : memref<128x128xf32>
    %50 = memref.alloc() : memref<2x128x128xf32>
    %51 = memref.alloc() : memref<128xf32>
    %52 = memref.alloc() : memref<128x128xf32>
    %53 = memref.alloc() : memref<2x128x128xf32>
    %54 = memref.alloc() : memref<2x2x128x64xf32>
    %55 = memref.alloc() : memref<2x2x128x64xf32>
    %56 = memref.alloc() : memref<2x2x128x128xf32>
    %57 = memref.alloc() : memref<2x2x128x64xf32>
    %58 = memref.alloc() : memref<2x2x128x128xf32>
    %59 = memref.alloc() : memref<2x2x128x64xf32>
    %60 = memref.alloc() : memref<2x128x2x64xf32>
    %61 = memref.alloc() : memref<128xf32>
    %62 = memref.alloc() : memref<128x128xf32>
    %63 = memref.alloc() : memref<2x128x128xf32>
    %64 = memref.alloc() : memref<2x128x128xf32>
    %65 = memref.alloc() : memref<128xf32>
    %66 = memref.alloc() : memref<128xf32>
    %67 = memref.alloc() : memref<2x128x128xf32>
    %68 = memref.alloc() : memref<2x128x128xf32>
    %69 = memref.alloc() : memref<512xf32>
    %70 = memref.alloc() : memref<512x128xf32>
    %71 = memref.alloc() : memref<2x128x128xf32>
    %72 = memref.alloc() : memref<128xf32>
    %73 = memref.alloc() : memref<128x512xf32>
    %74 = memref.alloc() : memref<2x128x512xf32>
    %75 = memref.alloc() : memref<2x128x128xf32>
    %76 = memref.alloc() : memref<128xf32>
    %77 = memref.alloc() : memref<128xf32>
    %78 = memref.alloc() : memref<2x128x128xf32>
    %79 = memref.alloc() : memref<128xf32>
    %80 = memref.alloc() : memref<128x128xf32>
    %81 = memref.alloc() : memref<2x128x128xf32>
    %82 = memref.alloc() : memref<128xf32>
    %83 = memref.alloc() : memref<128xf32>
    %84 = memref.alloc() : memref<2x128x128xf32>
    %85 = memref.alloc() : memref<256x30522xf32>
    %86 = memref.alloc() : memref<256x128xf32>
    %87 = memref.alloc() : memref<256xf32>
    %88 = memref.alloc() : memref<256xf32>
    %89 = memref.alloc() : memref<2x128x128xf32>
    %90 = memref.alloc() : memref<0xf32>
    %91 = memref.alloc() : memref<2x128x128xf32>
    %92 = memref.alloc() : memref<2x128x128xf32>
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<256xf32>
    %95 = memref.alloc() : memref<256xf32>
    %96 = memref.alloc() : memref<2x128x128xf32>
    %97 = memref.alloc() : memref<2x128x128xui8>
    %98 = memref.alloc() : memref<2x128x128xf32>
    %99 = memref.alloc() : memref<2x128x128xf32>
    %100 = memref.alloc() : memref<0xf32>
    %101 = memref.alloc() : memref<2x128x512xf32>
    %102 = memref.alloc() : memref<2x128x512xf32>
    %103 = memref.alloc() : memref<2x128x128xf32>
    %104 = memref.alloc() : memref<256xf32>
    %105 = memref.alloc() : memref<256xf32>
    %106 = memref.alloc() : memref<2x128x128xf32>
    %107 = memref.alloc() : memref<2x128x128xui8>
    %108 = memref.alloc() : memref<2x128x128xf32>
    %109 = memref.alloc() : memref<2x128x128xf32>
    %110 = memref.alloc() : memref<2x128x128xf32>
    %111 = memref.alloc() : memref<2x128x2x64xf32>
    %112 = memref.alloc() : memref<2x2x128x64xf32>
    %113 = memref.alloc() : memref<2x2x128x64xf32>
    %114 = memref.alloc() : memref<2x2x128x128xui8>
    %115 = memref.alloc() : memref<2x2x128x128xf32>
    %116 = memref.alloc() : memref<2x2x128x128xf32>
    %117 = memref.alloc() : memref<2x2x128x128xf32>
    %118 = memref.alloc() : memref<2x2x128x64xf32>
    %119 = memref.alloc() : memref<2x2x128x64xf32>
    %120 = memref.alloc() : memref<2x128x128xf32>
    %121 = memref.alloc() : memref<256xf32>
    %122 = memref.alloc() : memref<256xf32>
    %123 = memref.alloc() : memref<2x128x128xf32>
    %124 = memref.alloc() : memref<2x128x128xui8>
    %125 = memref.alloc() : memref<2x128x128xf32>
    %126 = memref.alloc() : memref<2x128x128xf32>
    %127 = memref.alloc() : memref<0xf32>
    %128 = memref.alloc() : memref<2x128x512xf32>
    %129 = memref.alloc() : memref<2x128x512xf32>
    %130 = memref.alloc() : memref<2x128x128xf32>
    %131 = memref.alloc() : memref<256xf32>
    %132 = memref.alloc() : memref<256xf32>
    %133 = memref.alloc() : memref<2x128x128xf32>
    %134 = memref.alloc() : memref<2x128x128xui8>
    %135 = memref.alloc() : memref<2x128x128xf32>
    %136 = memref.alloc() : memref<2x128x128xf32>
    %137 = memref.alloc() : memref<2x128x128xf32>
    %138 = memref.alloc() : memref<2x128x2x64xf32>
    %139 = memref.alloc() : memref<2x2x128x64xf32>
    %140 = memref.alloc() : memref<2x2x128x64xf32>
    %141 = memref.alloc() : memref<2x2x128x128xui8>
    %142 = memref.alloc() : memref<2x2x128x128xf32>
    %143 = memref.alloc() : memref<2x2x128x128xf32>
    %144 = memref.alloc() : memref<2x2x128x128xf32>
    %145 = memref.alloc() : memref<2x2x128x64xf32>
    %146 = memref.alloc() : memref<2x2x128x64xf32>
    %147 = memref.alloc() : memref<2x128x128xf32>
    %148 = memref.alloc() : memref<256xf32>
    %149 = memref.alloc() : memref<256xf32>
    %150 = memref.alloc() : memref<2x128x128xf32>
    %151 = memref.alloc() : memref<1x128x128xf32>
    %152 = memref.alloc() : memref<128x128xf32>
    %153 = memref.alloc() : memref<256x128xf32>
    %154 = memref.alloc() : memref<2x128x128xf32>
    %155 = memref.alloc() : memref<256x128xf32>
    %156 = memref.alloc() : memref<256x128xf32>
    %157 = memref.alloc() : memref<256x30522xf32>
    %158 = memref.alloc() : memref<1x128xi64>
    %159 = memref.alloc() : memref<128xi64>
    %160 = memref.alloc() : memref<1x128xi64>
    %161 = memref.alloc() : memref<f32>
    %162 = memref.alloc() : memref<128x128xf32>
    %163 = memref.alloc() : memref<128xf64>
    %164 = memref.alloc() : memref<128xi64>
    %165 = memref.alloc() : memref<128xi64>
    %166 = memref.alloc() : memref<512x128xf32>
    %167 = memref.alloc() : memref<256x128xf32>
    %168 = memref.alloc() : memref<256xf64>
    %169 = memref.alloc() : memref<256xi64>
    %170 = memref.alloc() : memref<256xi64>
    %171 = memref.alloc() : memref<2x128xf32>
    %172 = memref.alloc() : memref<256xf64>
    %173 = memref.alloc() : memref<256xi64>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<30522x128xf32>} : (memref<30522x128xf32>) -> ()
    "lmhlo.constant"(%173) {value = dense<30522> : tensor<256xi64>} : (memref<256xi64>) -> ()
    "lmhlo.constant"(%172) {value = dense<0.000000e+00> : tensor<256xf64>} : (memref<256xf64>) -> ()
    "lmhlo.constant"(%171) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    "lmhlo.constant"(%170) {value = dense<0> : tensor<256xi64>} : (memref<256xi64>) -> ()
    "lmhlo.constant"(%169) {value = dense<2> : tensor<256xi64>} : (memref<256xi64>) -> ()
    "lmhlo.constant"(%168) {value = dense<-1.000000e+00> : tensor<256xf64>} : (memref<256xf64>) -> ()
    "lmhlo.constant"(%167) {value = dense<0.000000e+00> : tensor<256x128xf32>} : (memref<256x128xf32>) -> ()
    "lmhlo.constant"(%166) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    "lmhlo.constant"(%165) {value = dense<0> : tensor<128xi64>} : (memref<128xi64>) -> ()
    "lmhlo.constant"(%164) {value = dense<512> : tensor<128xi64>} : (memref<128xi64>) -> ()
    "lmhlo.constant"(%163) {value = dense<-1.000000e+00> : tensor<128xf64>} : (memref<128xf64>) -> ()
    "lmhlo.constant"(%162) {value = dense<0.000000e+00> : tensor<128x128xf32>} : (memref<128x128xf32>) -> ()
    "lmhlo.constant"(%161) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.slice"(%arg1, %160) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.reshape"(%160, %159) : (memref<1x128xi64>, memref<128xi64>) -> ()
    "lmhlo.slice"(%arg2, %158) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.reshape"(%arg47, %157) : (memref<2x128x30522xf32>, memref<256x30522xf32>) -> ()
    %174:3 = call @Unknown0(%arg0, %170, %173, %172) : (memref<2x128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg3, %174#0, %156) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    "lmhlo.dot"(%157, %arg3, %155) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.reshape"(%155, %154) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    %175:3 = call @Unknown1(%159, %170, %169, %168) : (memref<128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg4, %175#0, %153) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %176 = call @Unknown2(%156, %153) : (memref<256x128xf32>, memref<256x128xf32>) -> memref<2x128x128xf32>
    %177:3 = call @Unknown3(%158, %165, %164, %163) : (memref<1x128xi64>, memref<128xi64>, memref<128xi64>, memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    "lmhlo.gather"(%arg5, %177#0, %152) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    "lmhlo.reshape"(%152, %151) : (memref<128x128xf32>, memref<1x128x128xf32>) -> ()
    "lmhlo.custom_call"(%176, %arg6, %arg7, %151, %150, %149, %148, %147) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%150, %arg8, %arg9, %146) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%150, %arg10, %arg11, %145) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%146, %145, %144) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%144, %arg14, %143, %142, %141) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%150, %arg12, %arg13, %140) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%142, %140, %139) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%139, %138) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%138, %137) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%137, %arg15, %arg16, %136, %135, %134) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%136, %arg17, %arg18, %150, %133, %132, %131, %130) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%133, %arg19, %arg20, %129, %128, %127) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%129, %arg21, %arg22, %126, %125, %124) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%126, %arg23, %arg24, %133, %123, %122, %121, %120) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%123, %arg25, %arg26, %119) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%123, %arg27, %arg28, %118) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%119, %118, %117) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%117, %arg31, %116, %115, %114) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%123, %arg29, %arg30, %113) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%115, %113, %112) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%112, %111) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%111, %110) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%110, %arg32, %arg33, %109, %108, %107) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%109, %arg34, %arg35, %123, %106, %105, %104, %103) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%106, %arg36, %arg37, %102, %101, %100) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%102, %arg38, %arg39, %99, %98, %97) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%99, %arg40, %arg41, %106, %96, %95, %94, %93) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%96, %arg42, %arg43, %92, %91, %90) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%92, %arg44, %arg45, %89, %88, %87) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.reshape"(%89, %86) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.dot"(%86, %arg3, %85) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %178 = call @Unknown4(%85, %arg46) : (memref<256x30522xf32>, memref<30522xf32>) -> memref<2x128x30522xf32>
    %179 = call @MatmulOp0(%86, %157) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    "lmhlo.custom_call"(%154, %92, %arg44, %88, %87, %84, %83, %82) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%84, %96, %arg42, %91, %90, %81, %80, %79) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%81, %93, %arg40, %95, %94, %78, %77, %76, %75) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%78, %102, %arg38, %98, %97, %74, %73, %72) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%74, %106, %arg36, %101, %100, %71, %70, %69) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    "lmhlo.add"(%75, %71, %68) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%68, %103, %arg34, %105, %104, %67, %66, %65, %64) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%67, %110, %arg32, %108, %107, %63, %62, %61) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%63, %60) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%60, %59) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%59, %115, %113, %58, %57) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%58, %116, %114, %56) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%56, %119, %118, %55, %54) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%55, %123, %arg25, %53, %52, %51) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%57, %123, %arg29, %50, %49, %48) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%54, %123, %arg27, %47, %46, %45) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %180 = call @Unknown5(%64, %53, %50, %47) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%180, %120, %arg23, %122, %121, %44, %43, %42, %41) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%44, %129, %arg21, %125, %124, %40, %39, %38) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%40, %133, %arg19, %128, %127, %37, %36, %35) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    "lmhlo.add"(%41, %37, %34) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%34, %130, %arg17, %132, %131, %33, %32, %31, %30) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%33, %137, %arg15, %135, %134, %29, %28, %27) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%29, %26) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%26, %25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%25, %142, %140, %24, %23) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%24, %143, %141, %22) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%22, %146, %145, %21, %20) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%21, %150, %arg8, %19, %18, %17) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%23, %150, %arg12, %16, %15, %14) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%20, %150, %arg10, %13, %12, %11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %181 = call @Unknown6(%30, %19, %16, %13) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%181, %147, %arg6, %149, %148, %10, %9, %8, %7) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %182:2 = call @Unknown7(%174#2, %10, %167, %175#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    "lmhlo.scatter"(%0, %174#1, %182#0, %6) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %184 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%184) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    "lmhlo.add"(%179, %6, %5) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (memref<30522x128xf32>, memref<30522x128xf32>, memref<30522x128xf32>) -> ()
    "lmhlo.scatter"(%171, %175#1, %182#1, %4) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %184 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%184) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    "lmhlo.reduce"(%7, %161, %3) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %183 = call @Unknown8(%177#2, %3, %162) : (memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>) -> memref<128x128xf32>
    "lmhlo.scatter"(%166, %177#1, %183, %2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %184 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%184) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    "lmhlo.reduce"(%arg47, %161, %1) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %178, %5, %4, %2, %9, %8, %18, %17, %12, %11, %15, %14, %28, %27, %32, %31, %36, %35, %39, %38, %43, %42, %52, %51, %46, %45, %49, %48, %62, %61, %66, %65, %70, %69, %73, %72, %77, %76, %80, %79, %83, %82, %1 : memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}
