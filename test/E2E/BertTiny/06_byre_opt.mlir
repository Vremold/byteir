// RUN: byteir-opt %s -byre-opt="append-arg-types" | FileCheck %s

// CHECK-LABEL: func @main
module attributes {gpu.container_module} {
  func private @MatmulOp0(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<128x30522xf32>
    %1 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %1 : memref<30522x128xf32>
  }
  func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [2 : i32, 1 : i32], __byre__kernel_name = "Unknown0_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, passthrough_arg = [1 : i32, 0 : i32]} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xi1>
    %1 = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    gpu.launch_func  @Unknown0_kernel::@Unknown0_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128xi64>, %0 : memref<256xi1>)
    return %1, %0 : memref<256xi64>, memref<256xi1>
  }
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
  func private @Unknown1(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown1_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xi1>
    %1 = memref.alloc() : memref<256xi64>
    %2 = memref.expand_shape %1 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %3 = memref.alloc() : memref<256xui32>
    gpu.launch_func  @Unknown1_kernel::@Unknown1_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128xi64>, %3 : memref<256xui32>, %1 : memref<256xi64>, %0 : memref<256xi1>)
    return %3, %2, %0 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
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
  func private @Unknown2(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown2_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<2x128xi1>
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %2 = memref.alloc() : memref<2x128xi64>
    %3 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %4 = memref.expand_shape %3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %5 = memref.alloc() : memref<2x128xui32>
    %6 = memref.collapse_shape %5 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    gpu.launch_func  @Unknown2_kernel::@Unknown2_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xi64>, %5 : memref<2x128xui32>, %2 : memref<2x128xi64>, %0 : memref<2x128xi1>)
    return %6, %4, %1 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
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
  func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown3_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
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
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %c512_i64 = arith.constant 512 : i64
      %c0_i64 = arith.constant 0 : i64
      %cst = arith.constant -1.000000e+00 : f64
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xi64>
        %7 = arith.trunci %6 : i64 to i32
        %8 = builtin.unrealized_conversion_cast %7 : i32 to ui32
        memref.store %8, %arg1[%4] : memref<128xui32>
        %9 = arith.addi %6, %c512_i64 : i64
        %10 = arith.cmpi slt, %6, %c0_i64 : i64
        %11 = select %10, %9, %6 : i64
        memref.store %11, %arg2[%4] : memref<128xi64>
        %12 = arith.sitofp %6 : i64 to f64
        %13 = arith.cmpf une, %12, %cst : f64
        memref.store %13, %arg3[%4] : memref<128xi1>
      }
      gpu.return
    }
  }
  func private @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], __byre__kernel_name = "Unknown4_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    gpu.launch_func  @Unknown4_kernel::@Unknown4_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x128xf32>, %arg1 : memref<256x128xf32>, %arg2 : memref<128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
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
  func private @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 244176 : i32, __byre__arg_ranks = [2 : i32, 1 : i32, 3 : i32, 3 : i32, 2 : i32], __byre__kernel_name = "Unknown5_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 0 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c244176 = arith.constant 244176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    %1 = memref.alloc() : memref<256x30522xf32>
    %2 = memref.alloc() : memref<2x128x30522xf32>
    gpu.launch_func  @Unknown5_kernel::@Unknown5_kernel blocks in (%c244176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x30522xf32>, %arg1 : memref<30522xf32>, %2 : memref<2x128x30522xf32>, %0 : memref<2x128x30522xf32>, %1 : memref<256x30522xf32>)
    return %2, %1 : memref<2x128x30522xf32>, memref<256x30522xf32>
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
  func private @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 244176 : i32, __byre__arg_ranks = [2 : i32, 1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown6_kernel", __byteir_elementwise_fusion__, arg_offsets = [1 : i32, 0 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c244176 = arith.constant 244176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x30522xf32>
    %1 = memref.alloc() : memref<256x30522xf32>
    gpu.launch_func  @Unknown6_kernel::@Unknown6_kernel blocks in (%c244176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg1 : memref<256x30522xf32>, %arg0 : memref<256xf32>, %0 : memref<256x30522xf32>, %1 : memref<256x30522xf32>)
    return %0, %1 : memref<256x30522xf32>, memref<256x30522xf32>
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
  func private @Unknown7(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown7_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown7_kernel::@Unknown7_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
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
  func private @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 244176 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 2 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown8_kernel", __byteir_elementwise_fusion__, arg_offsets = [3 : i32, 2 : i32, 4 : i32, 1 : i32, 0 : i32, 5 : i32, 6 : i32, 7 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c244176 = arith.constant 244176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x30522xf32>
    %1 = memref.alloc() : memref<256x30522xf32>
    %2 = memref.alloc() : memref<256x30522xf32>
    %3 = memref.alloc() : memref<256x30522xf32>
    gpu.launch_func  @Unknown8_kernel::@Unknown8_kernel blocks in (%c244176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg3 : memref<256xi1>, %arg2 : memref<256xi64>, %3 : memref<256x30522xf32>, %arg1 : memref<256x30522xf32>, %arg0 : memref<256xf32>, %2 : memref<256x30522xf32>, %1 : memref<256x30522xf32>, %0 : memref<256x30522xf32>)
    return %3, %2, %1, %0 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
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
  func private @Unknown9(%arg0: memref<f32>) -> memref<f32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [0 : i32, 0 : i32], __byre__kernel_name = "Unknown9_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<f32>
    gpu.launch_func  @Unknown9_kernel::@Unknown9_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<f32>, %0 : memref<f32>)
    return %0 : memref<f32>
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
  func private @Unknown10(%arg0: memref<f32>, %arg1: memref<256x30522xf32>) -> memref<256x30522xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 244176 : i32, __byre__arg_ranks = [2 : i32, 0 : i32, 2 : i32], __byre__kernel_name = "Unknown10_kernel", __byteir_elementwise_fusion__, arg_offsets = [1 : i32, 0 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c244176 = arith.constant 244176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x30522xf32>
    gpu.launch_func  @Unknown10_kernel::@Unknown10_kernel blocks in (%c244176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg1 : memref<256x30522xf32>, %arg0 : memref<f32>, %0 : memref<256x30522xf32>)
    return %0 : memref<256x30522xf32>
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
  func private @Unknown11(%arg0: memref<f32>, %arg1: memref<f32>) -> memref<f32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [0 : i32, 0 : i32, 0 : i32], __byre__kernel_name = "Unknown11_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<f32>
    gpu.launch_func  @Unknown11_kernel::@Unknown11_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<f32>, %arg1 : memref<f32>, %0 : memref<f32>)
    return %0 : memref<f32>
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
  func private @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 244176 : i32, __byre__arg_ranks = [2 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 3 : i32], __byre__kernel_name = "Unknown12_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 1 : i32, 0 : i32, 3 : i32, 0 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c244176 = arith.constant 244176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x30522xf32>
    %1 = memref.alloc() : memref<2x128x30522xf32>
    %2 = memref.expand_shape %arg0 [[0, 1]] : memref<256xf32> into memref<2x128xf32>
    gpu.launch_func  @Unknown12_kernel::@Unknown12_kernel blocks in (%c244176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<256x30522xf32>, %arg1 : memref<256x30522xf32>, %arg0 : memref<256xf32>, %0 : memref<256x30522xf32>, %2 : memref<2x128xf32>, %1 : memref<2x128x30522xf32>)
    return %0, %1 : memref<256x30522xf32>, memref<2x128x30522xf32>
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
  func private @Unknown13(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown13_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    gpu.launch_func  @Unknown13_kernel::@Unknown13_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
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
  func private @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown14_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    gpu.launch_func  @Unknown14_kernel::@Unknown14_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %arg2 : memref<2x128x128xf32>, %arg3 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
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
  func private @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown15_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    gpu.launch_func  @Unknown15_kernel::@Unknown15_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
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
  func private @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown16_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    gpu.launch_func  @Unknown16_kernel::@Unknown16_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %arg2 : memref<2x128x128xf32>, %arg3 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
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
  func private @Unknown17(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [2 : i32, 3 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown17_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 3 : i32, 2 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x128xf32>
    %1 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %2 = memref.alloc() : memref<256x128xf32>
    %3 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    gpu.launch_func  @Unknown17_kernel::@Unknown17_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%3 : memref<2x128xi1>, %arg1 : memref<2x128x128xf32>, %2 : memref<256x128xf32>, %1 : memref<2x128xi1>, %0 : memref<256x128xf32>)
    return %2, %0 : memref<256x128xf32>, memref<256x128xf32>
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
  func private @Unknown18(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 512 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown18_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x128xf32>
    gpu.launch_func  @Unknown18_kernel::@Unknown18_kernel blocks in (%c512, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xi1>, %arg1 : memref<128x128xf32>, %0 : memref<128x128xf32>)
    return %0 : memref<128x128xf32>
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
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<2x128xi64>, %arg2: memref<1x512xi64>, %arg3: memref<1x512xi64>, %arg4: memref<30522x128xf32>, %arg5: memref<2x128xf32>, %arg6: memref<512x128xf32>, %arg7: memref<128xf32>, %arg8: memref<128xf32>, %arg9: memref<128x128xf32>, %arg10: memref<128xf32>, %arg11: memref<128x128xf32>, %arg12: memref<128xf32>, %arg13: memref<128x128xf32>, %arg14: memref<128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<512x128xf32>, %arg36: memref<512xf32>, %arg37: memref<128x512xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128x128xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<f32>
    %1 = memref.alloc() : memref<30522xf32>
    %2 = memref.alloc() : memref<512x128xf32>
    %3 = memref.alloc() : memref<128x128xf32>
    %4 = memref.alloc() : memref<2x128xf32>
    %5 = memref.alloc() : memref<30522x128xf32>
    %6 = memref.alloc() : memref<128xf32>
    %7 = memref.alloc() : memref<128xf32>
    %8 = memref.alloc() : memref<2x128x128xf32>
    %9 = memref.alloc() : memref<128xf32>
    %10 = memref.alloc() : memref<128x128xf32>
    %11 = memref.alloc() : memref<2x128x128xf32>
    %12 = memref.alloc() : memref<128xf32>
    %13 = memref.alloc() : memref<128x128xf32>
    %14 = memref.alloc() : memref<2x128x128xf32>
    %15 = memref.alloc() : memref<128xf32>
    %16 = memref.alloc() : memref<128x128xf32>
    %17 = memref.alloc() : memref<2x128x128xf32>
    %18 = memref.alloc() : memref<2x2x128x64xf32>
    %19 = memref.alloc() : memref<2x2x128x64xf32>
    %20 = memref.alloc() : memref<2x2x128x128xf32>
    %21 = memref.alloc() : memref<2x2x128x64xf32>
    %22 = memref.alloc() : memref<2x2x128x128xf32>
    %23 = memref.alloc() : memref<2x2x128x64xf32>
    %24 = memref.alloc() : memref<2x128x2x64xf32>
    %25 = memref.alloc() : memref<128xf32>
    %26 = memref.alloc() : memref<128x128xf32>
    %27 = memref.alloc() : memref<2x128x128xf32>
    %28 = memref.alloc() : memref<2x128x128xf32>
    %29 = memref.alloc() : memref<128xf32>
    %30 = memref.alloc() : memref<128xf32>
    %31 = memref.alloc() : memref<2x128x128xf32>
    %32 = memref.alloc() : memref<512xf32>
    %33 = memref.alloc() : memref<512x128xf32>
    %34 = memref.alloc() : memref<2x128x128xf32>
    %35 = memref.alloc() : memref<128xf32>
    %36 = memref.alloc() : memref<128x512xf32>
    %37 = memref.alloc() : memref<2x128x512xf32>
    %38 = memref.alloc() : memref<2x128x128xf32>
    %39 = memref.alloc() : memref<128xf32>
    %40 = memref.alloc() : memref<128xf32>
    %41 = memref.alloc() : memref<2x128x128xf32>
    %42 = memref.alloc() : memref<128xf32>
    %43 = memref.alloc() : memref<128x128xf32>
    %44 = memref.alloc() : memref<2x128x128xf32>
    %45 = memref.alloc() : memref<128xf32>
    %46 = memref.alloc() : memref<128x128xf32>
    %47 = memref.alloc() : memref<2x128x128xf32>
    %48 = memref.alloc() : memref<128xf32>
    %49 = memref.alloc() : memref<128x128xf32>
    %50 = memref.alloc() : memref<2x128x128xf32>
    %51 = memref.alloc() : memref<2x2x128x64xf32>
    %52 = memref.alloc() : memref<2x2x128x64xf32>
    %53 = memref.alloc() : memref<2x2x128x128xf32>
    %54 = memref.alloc() : memref<2x2x128x64xf32>
    %55 = memref.alloc() : memref<2x2x128x128xf32>
    %56 = memref.alloc() : memref<2x2x128x64xf32>
    %57 = memref.alloc() : memref<2x128x2x64xf32>
    %58 = memref.alloc() : memref<128xf32>
    %59 = memref.alloc() : memref<128x128xf32>
    %60 = memref.alloc() : memref<2x128x128xf32>
    %61 = memref.alloc() : memref<2x128x128xf32>
    %62 = memref.alloc() : memref<128xf32>
    %63 = memref.alloc() : memref<128xf32>
    %64 = memref.alloc() : memref<2x128x128xf32>
    %65 = memref.alloc() : memref<512xf32>
    %66 = memref.alloc() : memref<512x128xf32>
    %67 = memref.alloc() : memref<2x128x128xf32>
    %68 = memref.alloc() : memref<128xf32>
    %69 = memref.alloc() : memref<128x512xf32>
    %70 = memref.alloc() : memref<2x128x512xf32>
    %71 = memref.alloc() : memref<2x128x128xf32>
    %72 = memref.alloc() : memref<128xf32>
    %73 = memref.alloc() : memref<128xf32>
    %74 = memref.alloc() : memref<2x128x128xf32>
    %75 = memref.alloc() : memref<128xf32>
    %76 = memref.alloc() : memref<128x128xf32>
    %77 = memref.alloc() : memref<2x128x128xf32>
    %78 = memref.alloc() : memref<128xf32>
    %79 = memref.alloc() : memref<128xf32>
    %80 = memref.alloc() : memref<2x128x128xf32>
    %81 = memref.alloc() : memref<2x128x128xf32>
    %82 = memref.alloc() : memref<256x128xf32>
    %83 = memref.alloc() : memref<f32>
    %84 = memref.alloc() : memref<256xf32>
    %85 = memref.alloc() : memref<f32>
    %86 = memref.alloc() : memref<f32>
    %87 = memref.alloc() : memref<256xf32>
    %88 = memref.alloc() : memref<256xf32>
    %89 = memref.alloc() : memref<256x30522xf32>
    %90 = memref.alloc() : memref<256x128xf32>
    %91 = memref.alloc() : memref<256xf32>
    %92 = memref.alloc() : memref<256xf32>
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<0xf32>
    %95 = memref.alloc() : memref<2x128x128xf32>
    %96 = memref.alloc() : memref<2x128x128xf32>
    %97 = memref.alloc() : memref<2x128x128xf32>
    %98 = memref.alloc() : memref<256xf32>
    %99 = memref.alloc() : memref<256xf32>
    %100 = memref.alloc() : memref<2x128x128xf32>
    %101 = memref.alloc() : memref<2x128x128xf32>
    %102 = memref.alloc() : memref<0xf32>
    %103 = memref.alloc() : memref<2x128x512xf32>
    %104 = memref.alloc() : memref<2x128x512xf32>
    %105 = memref.alloc() : memref<2x128x128xf32>
    %106 = memref.alloc() : memref<256xf32>
    %107 = memref.alloc() : memref<256xf32>
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
    %124 = memref.alloc() : memref<2x128x128xf32>
    %125 = memref.alloc() : memref<0xf32>
    %126 = memref.alloc() : memref<2x128x512xf32>
    %127 = memref.alloc() : memref<2x128x512xf32>
    %128 = memref.alloc() : memref<2x128x128xf32>
    %129 = memref.alloc() : memref<256xf32>
    %130 = memref.alloc() : memref<256xf32>
    %131 = memref.alloc() : memref<2x128x128xf32>
    %132 = memref.alloc() : memref<2x128x128xf32>
    %133 = memref.alloc() : memref<2x128x128xf32>
    %134 = memref.alloc() : memref<2x128x2x64xf32>
    %135 = memref.alloc() : memref<2x2x128x64xf32>
    %136 = memref.alloc() : memref<2x2x128x64xf32>
    %137 = memref.alloc() : memref<2x2x128x128xui8>
    %138 = memref.alloc() : memref<2x2x128x128xf32>
    %139 = memref.alloc() : memref<2x2x128x128xf32>
    %140 = memref.alloc() : memref<2x2x128x128xf32>
    %141 = memref.alloc() : memref<2x2x128x64xf32>
    %142 = memref.alloc() : memref<2x2x128x64xf32>
    %143 = memref.alloc() : memref<256xf32>
    %144 = memref.alloc() : memref<256xf32>
    %145 = memref.alloc() : memref<2x128x128xf32>
    %146 = memref.alloc() : memref<128x128xf32>
    %147 = memref.alloc() : memref<256x128xf32>
    %148 = memref.alloc() : memref<256x128xf32>
    %149 = memref.alloc() : memref<1x128xi64>
    %150 = memref.alloc() : memref<128xi64>
    %151 = memref.alloc() : memref<1x128xi64>
    %152 = memref.alloc() : memref<2x128x128xf32>
    %153 = memref.alloc() : memref<f32>
    %154 = memref.alloc() : memref<2x128xf32>
    %155 = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%155) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    "lmhlo.constant"(%154) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    "lmhlo.constant"(%153) {value = dense<0xFF800000> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%152) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : (memref<2x128x128xf32>) -> ()
    "lmhlo.slice"(%arg2, %151) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.reshape"(%151, %150) : (memref<1x128xi64>, memref<128xi64>) -> ()
    "lmhlo.slice"(%arg3, %149) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %156:2 = call @Unknown0(%arg1) : (memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>)
    %157:3 = call @Unknown1(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg4, %157#0, %148) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %158:3 = call @Unknown2(%150) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg5, %158#0, %147) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %159:3 = call @Unknown3(%149) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    "lmhlo.gather"(%arg6, %159#0, %146) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %160 = call @Unknown4(%148, %147, %146) : (memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%160, %arg7, %arg8, %145, %144, %143) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.custom_call"(%145, %arg9, %arg10, %142) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%145, %arg11, %arg12, %141) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%142, %141, %140) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%140, %152, %139, %138, %137) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%145, %arg13, %arg14, %136) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%139, %136, %135) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%135, %134) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%134, %133) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%133, %arg15, %arg16, %132) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%132, %arg17, %arg18, %145, %131, %130, %129, %128) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%131, %arg19, %arg20, %127, %126, %125) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%127, %arg21, %arg22, %124) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%124, %arg23, %arg24, %131, %123, %122, %121, %120) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%123, %arg25, %arg26, %119) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%123, %arg27, %arg28, %118) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%119, %118, %117) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%117, %152, %116, %115, %114) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%123, %arg29, %arg30, %113) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%116, %113, %112) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%112, %111) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%111, %110) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%110, %arg31, %arg32, %109) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%109, %arg33, %arg34, %123, %108, %107, %106, %105) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%108, %arg35, %arg36, %104, %103, %102) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%104, %arg37, %arg38, %101) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%101, %arg39, %arg40, %108, %100, %99, %98, %97) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%100, %arg41, %arg42, %96, %95, %94) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%96, %arg43, %arg44, %93, %92, %91) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.reshape"(%93, %90) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.dot"(%90, %arg4, %89) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %161:2 = call @Unknown5(%89, %arg45) : (memref<256x30522xf32>, memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>)
    "lmhlo.reduce"(%161#1, %153, %88) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.maximum"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %162:2 = call @Unknown6(%88, %161#1) : (memref<256xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>)
    "lmhlo.reduce"(%162#1, %0, %87) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %163 = call @Unknown7(%87) : (memref<256xf32>) -> memref<256xf32>
    %164:4 = call @Unknown8(%163, %162#0, %156#0, %156#1) : (memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>)
    "lmhlo.reduce"(%164#0, %0, %86) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.reduce"(%164#0, %0, %85) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %165 = call @Unknown9(%85) : (memref<f32>) -> memref<f32>
    %166 = call @Unknown10(%165, %164#2) : (memref<f32>, memref<256x30522xf32>) -> memref<256x30522xf32>
    "lmhlo.reduce"(%166, %0, %84) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    "lmhlo.reduce"(%164#1, %0, %83) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %167 = call @Unknown11(%83, %86) : (memref<f32>, memref<f32>) -> memref<f32>
    %168:2 = call @Unknown12(%84, %164#3, %166) : (memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>)
    %169 = call @MatmulOp0(%90, %168#0) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    "lmhlo.dot"(%168#0, %arg4, %82) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.reshape"(%82, %81) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%81, %96, %arg43, %92, %91, %80, %79, %78) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%80, %100, %arg41, %95, %94, %77, %76, %75) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%77, %97, %arg39, %99, %98, %74, %73, %72, %71) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%74, %104, %arg37, %70, %69, %68) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%70, %108, %arg35, %103, %102, %67, %66, %65) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %170 = call @Unknown13(%71, %67) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%170, %105, %arg33, %107, %106, %64, %63, %62, %61) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%64, %110, %arg31, %60, %59, %58) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%60, %57) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%57, %56) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%56, %116, %113, %55, %54) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%55, %116, %114, %53) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%53, %119, %118, %52, %51) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%52, %123, %arg25, %50, %49, %48) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%54, %123, %arg29, %47, %46, %45) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%51, %123, %arg27, %44, %43, %42) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %171 = call @Unknown14(%61, %50, %47, %44) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%171, %120, %arg23, %122, %121, %41, %40, %39, %38) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%41, %127, %arg21, %37, %36, %35) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%37, %131, %arg19, %126, %125, %34, %33, %32) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %172 = call @Unknown15(%38, %34) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%172, %128, %arg17, %130, %129, %31, %30, %29, %28) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%31, %133, %arg15, %27, %26, %25) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%27, %24) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%24, %23) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%23, %139, %136, %22, %21) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%22, %139, %137, %20) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%20, %142, %141, %19, %18) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%19, %145, %arg9, %17, %16, %15) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%21, %145, %arg13, %14, %13, %12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%18, %145, %arg11, %11, %10, %9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %173 = call @Unknown16(%28, %17, %14, %11) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%173, %160, %arg7, %144, %143, %8, %7, %6) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %174:2 = call @Unknown17(%157#2, %8, %158#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    "lmhlo.scatter"(%169, %157#1, %174#0, %5) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    "lmhlo.scatter"(%154, %158#1, %174#1, %4) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    "lmhlo.reduce"(%8, %0, %3) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %175 = call @Unknown18(%159#2, %3) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    "lmhlo.scatter"(%155, %159#1, %175, %2) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):  // no predecessors
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    "lmhlo.reduce"(%168#1, %0, %1) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %161#0, %167, %5, %4, %2, %7, %6, %16, %15, %10, %9, %13, %12, %26, %25, %30, %29, %33, %32, %36, %35, %40, %39, %49, %48, %43, %42, %46, %45, %59, %58, %63, %62, %66, %65, %69, %68, %73, %72, %76, %75, %79, %78, %1 : memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

