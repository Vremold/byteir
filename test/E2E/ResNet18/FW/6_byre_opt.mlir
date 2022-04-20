// RUN: byteir-opt %s -byre-opt="append-arg-types" | FileCheck %s

// CHECK-LABEL: func @main
module attributes {gpu.container_module} {
  func private @Unknown0(%arg0: memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown0_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1176 = arith.constant 1176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x3x224x224xf16>
    gpu.launch_func  @Unknown0_kernel::@Unknown0_kernel blocks in (%c1176, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x3x224x224xf32>, %0 : memref<1x3x224x224xf16>)
    return %0 : memref<1x3x224x224xf16>
  }
  gpu.module @Unknown0_kernel {
    gpu.func @Unknown0_kernel(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x3x224x224xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c150528 = arith.constant 150528 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c150528 : index
      scf.if %5 {
        %c224 = arith.constant 224 : index
        %6 = arith.remsi %4, %c224 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c224 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c224 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c224 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c224 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c224 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c3 = arith.constant 3 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x3x224x224xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x3x224x224xf16>
      }
      gpu.return
    }
  }
  func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 74 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown1_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c74 = arith.constant 74 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x3x7x7xf16>
    gpu.launch_func  @Unknown1_kernel::@Unknown1_kernel blocks in (%c74, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x3x7x7xf32>, %0 : memref<64x3x7x7xf16>)
    return %0 : memref<64x3x7x7xf16>
  }
  gpu.module @Unknown1_kernel {
    gpu.func @Unknown1_kernel(%arg0: memref<64x3x7x7xf32>, %arg1: memref<64x3x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c9408 = arith.constant 9408 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c9408 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c3 = arith.constant 3 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x3x7x7xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x3x7x7xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp2(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x112x112xf32>
    %1 = memref.alloc() : memref<1x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x112x112xf32>, memref<1x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown3(%arg0: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 6272 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown3_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c6272 = arith.constant 6272 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x112x112xf16>
    gpu.launch_func  @Unknown3_kernel::@Unknown3_kernel blocks in (%c6272, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x112x112xf16>, %0 : memref<1x64x112x112xf16>)
    return %0 : memref<1x64x112x112xf16>
  }
  gpu.module @Unknown3_kernel {
    gpu.func @Unknown3_kernel(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c112 = arith.constant 112 : index
        %6 = arith.remsi %4, %c112 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c112 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c112 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c112 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c112 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c112 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x112x112xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
  }
  func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown4_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c288 = arith.constant 288 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown4_kernel::@Unknown4_kernel blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown4_kernel {
    gpu.func @Unknown4_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c36864 = arith.constant 36864 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp5(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown6(%arg0: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown6_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1568 = arith.constant 1568 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    gpu.launch_func  @Unknown6_kernel::@Unknown6_kernel blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %0 : memref<1x64x56x56xf16>)
    return %0 : memref<1x64x56x56xf16>
  }
  gpu.module @Unknown6_kernel {
    gpu.func @Unknown6_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown7(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown7_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c288 = arith.constant 288 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown7_kernel::@Unknown7_kernel blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown7_kernel {
    gpu.func @Unknown7_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c36864 = arith.constant 36864 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp8(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown9(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown9_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1568 = arith.constant 1568 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    gpu.launch_func  @Unknown9_kernel::@Unknown9_kernel blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %arg1 : memref<1x64x56x56xf16>, %0 : memref<1x64x56x56xf16>)
    return %0 : memref<1x64x56x56xf16>
  }
  gpu.module @Unknown9_kernel {
    gpu.func @Unknown9_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown10(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown10_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c288 = arith.constant 288 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown10_kernel::@Unknown10_kernel blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown10_kernel {
    gpu.func @Unknown10_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c36864 = arith.constant 36864 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp11(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown12(%arg0: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown12_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1568 = arith.constant 1568 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    gpu.launch_func  @Unknown12_kernel::@Unknown12_kernel blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %0 : memref<1x64x56x56xf16>)
    return %0 : memref<1x64x56x56xf16>
  }
  gpu.module @Unknown12_kernel {
    gpu.func @Unknown12_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown13(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 288 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown13_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c288 = arith.constant 288 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown13_kernel::@Unknown13_kernel blocks in (%c288, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown13_kernel {
    gpu.func @Unknown13_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c36864 = arith.constant 36864 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c36864 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp14(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown15(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1568 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown15_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1568 = arith.constant 1568 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    gpu.launch_func  @Unknown15_kernel::@Unknown15_kernel blocks in (%c1568, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x64x56x56xf16>, %arg1 : memref<1x64x56x56xf16>, %0 : memref<1x64x56x56xf16>)
    return %0 : memref<1x64x56x56xf16>
  }
  gpu.module @Unknown15_kernel {
    gpu.func @Unknown15_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c200704 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown16(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 64 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown16_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x1x1xf16>
    gpu.launch_func  @Unknown16_kernel::@Unknown16_kernel blocks in (%c64, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x64x1x1xf32>, %0 : memref<128x64x1x1xf16>)
    return %0 : memref<128x64x1x1xf16>
  }
  gpu.module @Unknown16_kernel {
    gpu.func @Unknown16_kernel(%arg0: memref<128x64x1x1xf32>, %arg1: memref<128x64x1x1xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c8192 = arith.constant 8192 : index
      %c0 = arith.constant 0 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c8192 : index
      scf.if %5 {
        %c64 = arith.constant 64 : index
        %6 = arith.remsi %4, %c64 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c64 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c64 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<128x64x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<128x64x1x1xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp17(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown18(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 576 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown18_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c576 = arith.constant 576 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x3x3xf16>
    gpu.launch_func  @Unknown18_kernel::@Unknown18_kernel blocks in (%c576, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x64x3x3xf32>, %0 : memref<128x64x3x3xf16>)
    return %0 : memref<128x64x3x3xf16>
  }
  gpu.module @Unknown18_kernel {
    gpu.func @Unknown18_kernel(%arg0: memref<128x64x3x3xf32>, %arg1: memref<128x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c73728 = arith.constant 73728 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c73728 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp19(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown20(%arg0: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown20_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    gpu.launch_func  @Unknown20_kernel::@Unknown20_kernel blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %0 : memref<1x128x28x28xf16>)
    return %0 : memref<1x128x28x28xf16>
  }
  gpu.module @Unknown20_kernel {
    gpu.func @Unknown20_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown21(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown21_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    gpu.launch_func  @Unknown21_kernel::@Unknown21_kernel blocks in (%c1152, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x128x3x3xf32>, %0 : memref<128x128x3x3xf16>)
    return %0 : memref<128x128x3x3xf16>
  }
  gpu.module @Unknown21_kernel {
    gpu.func @Unknown21_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c147456 = arith.constant 147456 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp22(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown23(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown23_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    gpu.launch_func  @Unknown23_kernel::@Unknown23_kernel blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %arg1 : memref<1x128x28x28xf16>, %0 : memref<1x128x28x28xf16>)
    return %0 : memref<1x128x28x28xf16>
  }
  gpu.module @Unknown23_kernel {
    gpu.func @Unknown23_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown24(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown24_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    gpu.launch_func  @Unknown24_kernel::@Unknown24_kernel blocks in (%c1152, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x128x3x3xf32>, %0 : memref<128x128x3x3xf16>)
    return %0 : memref<128x128x3x3xf16>
  }
  gpu.module @Unknown24_kernel {
    gpu.func @Unknown24_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c147456 = arith.constant 147456 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp25(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown26(%arg0: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown26_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    gpu.launch_func  @Unknown26_kernel::@Unknown26_kernel blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %0 : memref<1x128x28x28xf16>)
    return %0 : memref<1x128x28x28xf16>
  }
  gpu.module @Unknown26_kernel {
    gpu.func @Unknown26_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown27(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown27_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    gpu.launch_func  @Unknown27_kernel::@Unknown27_kernel blocks in (%c1152, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128x128x3x3xf32>, %0 : memref<128x128x3x3xf16>)
    return %0 : memref<128x128x3x3xf16>
  }
  gpu.module @Unknown27_kernel {
    gpu.func @Unknown27_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c147456 = arith.constant 147456 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c147456 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp28(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown29(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 784 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown29_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c784 = arith.constant 784 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    gpu.launch_func  @Unknown29_kernel::@Unknown29_kernel blocks in (%c784, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128x28x28xf16>, %arg1 : memref<1x128x28x28xf16>, %0 : memref<1x128x28x28xf16>)
    return %0 : memref<1x128x28x28xf16>
  }
  gpu.module @Unknown29_kernel {
    gpu.func @Unknown29_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c100352 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown30(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown30_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x1x1xf16>
    gpu.launch_func  @Unknown30_kernel::@Unknown30_kernel blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x128x1x1xf32>, %0 : memref<256x128x1x1xf16>)
    return %0 : memref<256x128x1x1xf16>
  }
  gpu.module @Unknown30_kernel {
    gpu.func @Unknown30_kernel(%arg0: memref<256x128x1x1xf32>, %arg1: memref<256x128x1x1xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<256x128x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<256x128x1x1xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown32(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown32_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2304 = arith.constant 2304 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x3x3xf16>
    gpu.launch_func  @Unknown32_kernel::@Unknown32_kernel blocks in (%c2304, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x128x3x3xf32>, %0 : memref<256x128x3x3xf16>)
    return %0 : memref<256x128x3x3xf16>
  }
  gpu.module @Unknown32_kernel {
    gpu.func @Unknown32_kernel(%arg0: memref<256x128x3x3xf32>, %arg1: memref<256x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c294912 = arith.constant 294912 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c294912 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp33(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown34(%arg0: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown34_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    gpu.launch_func  @Unknown34_kernel::@Unknown34_kernel blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %0 : memref<1x256x14x14xf16>)
    return %0 : memref<1x256x14x14xf16>
  }
  gpu.module @Unknown34_kernel {
    gpu.func @Unknown34_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown35(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown35_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    gpu.launch_func  @Unknown35_kernel::@Unknown35_kernel blocks in (%c4608, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x256x3x3xf32>, %0 : memref<256x256x3x3xf16>)
    return %0 : memref<256x256x3x3xf16>
  }
  gpu.module @Unknown35_kernel {
    gpu.func @Unknown35_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c589824 = arith.constant 589824 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp36(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown37(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown37_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    gpu.launch_func  @Unknown37_kernel::@Unknown37_kernel blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %arg1 : memref<1x256x14x14xf16>, %0 : memref<1x256x14x14xf16>)
    return %0 : memref<1x256x14x14xf16>
  }
  gpu.module @Unknown37_kernel {
    gpu.func @Unknown37_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown38(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown38_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    gpu.launch_func  @Unknown38_kernel::@Unknown38_kernel blocks in (%c4608, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x256x3x3xf32>, %0 : memref<256x256x3x3xf16>)
    return %0 : memref<256x256x3x3xf16>
  }
  gpu.module @Unknown38_kernel {
    gpu.func @Unknown38_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c589824 = arith.constant 589824 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp39(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown40(%arg0: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown40_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    gpu.launch_func  @Unknown40_kernel::@Unknown40_kernel blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %0 : memref<1x256x14x14xf16>)
    return %0 : memref<1x256x14x14xf16>
  }
  gpu.module @Unknown40_kernel {
    gpu.func @Unknown40_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown41(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown41_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    gpu.launch_func  @Unknown41_kernel::@Unknown41_kernel blocks in (%c4608, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x256x3x3xf32>, %0 : memref<256x256x3x3xf16>)
    return %0 : memref<256x256x3x3xf16>
  }
  gpu.module @Unknown41_kernel {
    gpu.func @Unknown41_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c589824 = arith.constant 589824 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c589824 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp42(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown43(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 392 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown43_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c392 = arith.constant 392 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    gpu.launch_func  @Unknown43_kernel::@Unknown43_kernel blocks in (%c392, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x256x14x14xf16>, %arg1 : memref<1x256x14x14xf16>, %0 : memref<1x256x14x14xf16>)
    return %0 : memref<1x256x14x14xf16>
  }
  gpu.module @Unknown43_kernel {
    gpu.func @Unknown43_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c50176 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown44(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown44_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x1x1xf16>
    gpu.launch_func  @Unknown44_kernel::@Unknown44_kernel blocks in (%c1024, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x256x1x1xf32>, %0 : memref<512x256x1x1xf16>)
    return %0 : memref<512x256x1x1xf16>
  }
  gpu.module @Unknown44_kernel {
    gpu.func @Unknown44_kernel(%arg0: memref<512x256x1x1xf32>, %arg1: memref<512x256x1x1xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c131072 = arith.constant 131072 : index
      %c0 = arith.constant 0 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c131072 : index
      scf.if %5 {
        %c256 = arith.constant 256 : index
        %6 = arith.remsi %4, %c256 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c256 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c256 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<512x256x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<512x256x1x1xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp45(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown46(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 9216 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown46_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c9216 = arith.constant 9216 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x3x3xf16>
    gpu.launch_func  @Unknown46_kernel::@Unknown46_kernel blocks in (%c9216, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x256x3x3xf32>, %0 : memref<512x256x3x3xf16>)
    return %0 : memref<512x256x3x3xf16>
  }
  gpu.module @Unknown46_kernel {
    gpu.func @Unknown46_kernel(%arg0: memref<512x256x3x3xf32>, %arg1: memref<512x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c1179648 = arith.constant 1179648 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1179648 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp47(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown48(%arg0: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown48_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    gpu.launch_func  @Unknown48_kernel::@Unknown48_kernel blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512x7x7xf16>, %0 : memref<1x512x7x7xf16>)
    return %0 : memref<1x512x7x7xf16>
  }
  gpu.module @Unknown48_kernel {
    gpu.func @Unknown48_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown49(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown49_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    gpu.launch_func  @Unknown49_kernel::@Unknown49_kernel blocks in (%c18432, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x512x3x3xf32>, %0 : memref<512x512x3x3xf16>)
    return %0 : memref<512x512x3x3xf16>
  }
  gpu.module @Unknown49_kernel {
    gpu.func @Unknown49_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c2359296 = arith.constant 2359296 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp50(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown51(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown51_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    gpu.launch_func  @Unknown51_kernel::@Unknown51_kernel blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512x7x7xf16>, %arg1 : memref<1x512x7x7xf16>, %0 : memref<1x512x7x7xf16>)
    return %0 : memref<1x512x7x7xf16>
  }
  gpu.module @Unknown51_kernel {
    gpu.func @Unknown51_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown52(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown52_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    gpu.launch_func  @Unknown52_kernel::@Unknown52_kernel blocks in (%c18432, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x512x3x3xf32>, %0 : memref<512x512x3x3xf16>)
    return %0 : memref<512x512x3x3xf16>
  }
  gpu.module @Unknown52_kernel {
    gpu.func @Unknown52_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c2359296 = arith.constant 2359296 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp53(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown54(%arg0: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown54_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    gpu.launch_func  @Unknown54_kernel::@Unknown54_kernel blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512x7x7xf16>, %0 : memref<1x512x7x7xf16>)
    return %0 : memref<1x512x7x7xf16>
  }
  gpu.module @Unknown54_kernel {
    gpu.func @Unknown54_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown55(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown55_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    gpu.launch_func  @Unknown55_kernel::@Unknown55_kernel blocks in (%c18432, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512x512x3x3xf32>, %0 : memref<512x512x3x3xf16>)
    return %0 : memref<512x512x3x3xf16>
  }
  gpu.module @Unknown55_kernel {
    gpu.func @Unknown55_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c2359296 = arith.constant 2359296 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c2359296 : index
      scf.if %5 {
        %c3 = arith.constant 3 : index
        %6 = arith.remsi %4, %c3 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c3 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp56(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown57(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 196 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown57_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c196 = arith.constant 196 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    gpu.launch_func  @Unknown57_kernel::@Unknown57_kernel blocks in (%c196, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512x7x7xf16>, %arg1 : memref<1x512x7x7xf16>, %0 : memref<1x512x7x7xf16>)
    return %0 : memref<1x512x7x7xf16>
  }
  gpu.module @Unknown57_kernel {
    gpu.func @Unknown57_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c25088 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = arith.select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = arith.select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = arith.select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown58(%arg0: memref<1x512xf16>) -> memref<1x512xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown58_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512xf16>
    gpu.launch_func  @Unknown58_kernel::@Unknown58_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x512xf16>, %0 : memref<1x512xf16>)
    return %0 : memref<1x512xf16>
  }
  gpu.module @Unknown58_kernel {
    gpu.func @Unknown58_kernel(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 2.040100e-02 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c512 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c512 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<1x512xf16>
        %17 = arith.mulf %16, %cst : f16
        memref.store %17, %arg1[%15, %9] : memref<1x512xf16>
      }
      gpu.return
    }
  }
  func private @Unknown59(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown59_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4000 = arith.constant 4000 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000x512xf16>
    gpu.launch_func  @Unknown59_kernel::@Unknown59_kernel blocks in (%c4000, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1000x512xf32>, %0 : memref<1000x512xf16>)
    return %0 : memref<1000x512xf16>
  }
  gpu.module @Unknown59_kernel {
    gpu.func @Unknown59_kernel(%arg0: memref<1000x512xf32>, %arg1: memref<1000x512xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512000 = arith.constant 512000 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512000 : index
      scf.if %5 {
        %c512 = arith.constant 512 : index
        %6 = arith.remsi %4, %c512 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c512 : index
        %9 = arith.select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<1000x512xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9] : memref<1000x512xf16>
      }
      gpu.return
    }
  }
  func private @Unknown60(%arg0: memref<1000xf32>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown60_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000xf16>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<1x1000xf16>
    gpu.launch_func  @Unknown60_kernel::@Unknown60_kernel blocks in (%c8, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1000xf32>, %0 : memref<1000xf16>, %arg1 : memref<1x1000xf16>, %1 : memref<1x1000xf16>)
    return %1 : memref<1x1000xf16>
  }
  gpu.module @Unknown60_kernel {
    gpu.func @Unknown60_kernel(%arg0: memref<1000xf32>, %arg1: memref<1000xf16>, %arg2: memref<1x1000xf16>, %arg3: memref<1x1000xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c1000 = arith.constant 1000 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1000 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<1000xf32>
        %7 = arith.truncf %6 : f32 to f16
        memref.store %7, %arg1[%4] : memref<1000xf16>
        %8 = arith.remsi %4, %c1000 : index
        %c0 = arith.constant 0 : index
        %9 = arith.cmpi slt, %8, %c0 : index
        %10 = arith.addi %8, %c1000 : index
        %11 = arith.select %9, %10, %8 : index
        %c-1 = arith.constant -1 : index
        %12 = arith.cmpi slt, %4, %c0 : index
        %13 = arith.subi %c-1, %4 : index
        %14 = arith.select %12, %13, %4 : index
        %15 = arith.divsi %14, %c1000 : index
        %16 = arith.subi %c-1, %15 : index
        %17 = arith.select %12, %16, %15 : index
        %18 = memref.load %arg2[%17, %11] : memref<1x1000xf16>
        %19 = memref.load %arg1[%11] : memref<1000xf16>
        %20 = arith.addf %18, %19 : f16
        memref.store %20, %arg3[%17, %11] : memref<1x1000xf16>
      }
      gpu.return
    }
  }
  func private @Unknown61(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown61_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown61_kernel::@Unknown61_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown61_kernel {
    gpu.func @Unknown61_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown62(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown62_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown62_kernel::@Unknown62_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown62_kernel {
    gpu.func @Unknown62_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown63(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown63_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown63_kernel::@Unknown63_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown63_kernel {
    gpu.func @Unknown63_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown64(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown64_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown64_kernel::@Unknown64_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown64_kernel {
    gpu.func @Unknown64_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown65(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown65_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown65_kernel::@Unknown65_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown65_kernel {
    gpu.func @Unknown65_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown66(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown66_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown66_kernel::@Unknown66_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown66_kernel {
    gpu.func @Unknown66_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown67(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown67_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown67_kernel::@Unknown67_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown67_kernel {
    gpu.func @Unknown67_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown68(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown68_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown68_kernel::@Unknown68_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown68_kernel {
    gpu.func @Unknown68_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown69(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown69_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown69_kernel::@Unknown69_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown69_kernel {
    gpu.func @Unknown69_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown70(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown70_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    gpu.launch_func  @Unknown70_kernel::@Unknown70_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown70_kernel {
    gpu.func @Unknown70_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c64 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<64xf32>
        %3 = memref.load %arg1[%0] : memref<64xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown71(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown71_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown71_kernel::@Unknown71_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown71_kernel {
    gpu.func @Unknown71_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown72(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown72_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown72_kernel::@Unknown72_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown72_kernel {
    gpu.func @Unknown72_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown73(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown73_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown73_kernel::@Unknown73_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown73_kernel {
    gpu.func @Unknown73_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown74(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown74_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown74_kernel::@Unknown74_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown74_kernel {
    gpu.func @Unknown74_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown75(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown75_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown75_kernel::@Unknown75_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown75_kernel {
    gpu.func @Unknown75_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown76(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown76_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown76_kernel::@Unknown76_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown76_kernel {
    gpu.func @Unknown76_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown77(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown77_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown77_kernel::@Unknown77_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown77_kernel {
    gpu.func @Unknown77_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown78(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown78_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown78_kernel::@Unknown78_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown78_kernel {
    gpu.func @Unknown78_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown79(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown79_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown79_kernel::@Unknown79_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown79_kernel {
    gpu.func @Unknown79_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown80(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown80_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    gpu.launch_func  @Unknown80_kernel::@Unknown80_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown80_kernel {
    gpu.func @Unknown80_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xf32>
        %3 = memref.load %arg1[%0] : memref<128xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.mulf %2, %cst_0 : f32
        %6 = arith.addf %5, %4 : f32
        memref.store %6, %arg2[%0] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown81(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown81_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown81_kernel::@Unknown81_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown81_kernel {
    gpu.func @Unknown81_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown82(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown82_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown82_kernel::@Unknown82_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown82_kernel {
    gpu.func @Unknown82_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown83(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown83_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown83_kernel::@Unknown83_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown83_kernel {
    gpu.func @Unknown83_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown84(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown84_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown84_kernel::@Unknown84_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown84_kernel {
    gpu.func @Unknown84_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown85(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown85_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown85_kernel::@Unknown85_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown85_kernel {
    gpu.func @Unknown85_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown86(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown86_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown86_kernel::@Unknown86_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown86_kernel {
    gpu.func @Unknown86_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown87(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown87_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown87_kernel::@Unknown87_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown87_kernel {
    gpu.func @Unknown87_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown88(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown88_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown88_kernel::@Unknown88_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown88_kernel {
    gpu.func @Unknown88_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown89(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown89_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown89_kernel::@Unknown89_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown89_kernel {
    gpu.func @Unknown89_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown90(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown90_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @Unknown90_kernel::@Unknown90_kernel blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown90_kernel {
    gpu.func @Unknown90_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = memref.load %arg1[%4] : memref<256xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @Unknown91(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown91_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown91_kernel::@Unknown91_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown91_kernel {
    gpu.func @Unknown91_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown92(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown92_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown92_kernel::@Unknown92_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown92_kernel {
    gpu.func @Unknown92_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown93(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown93_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown93_kernel::@Unknown93_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown93_kernel {
    gpu.func @Unknown93_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown94(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown94_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown94_kernel::@Unknown94_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown94_kernel {
    gpu.func @Unknown94_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown95(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown95_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown95_kernel::@Unknown95_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown95_kernel {
    gpu.func @Unknown95_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown96(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown96_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown96_kernel::@Unknown96_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown96_kernel {
    gpu.func @Unknown96_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown97(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown97_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown97_kernel::@Unknown97_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown97_kernel {
    gpu.func @Unknown97_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown98(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown98_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown98_kernel::@Unknown98_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown98_kernel {
    gpu.func @Unknown98_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown99(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown99_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown99_kernel::@Unknown99_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown99_kernel {
    gpu.func @Unknown99_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown100(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown100_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    gpu.launch_func  @Unknown100_kernel::@Unknown100_kernel blocks in (%c4, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown100_kernel {
    gpu.func @Unknown100_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = memref.load %arg1[%4] : memref<512xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<1000xf32>, %arg4: memref<1000x512xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64x64x3x3xf32>, %arg10: memref<64x64x3x3xf32>, %arg11: memref<64xf32>, %arg12: memref<64xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64x64x3x3xf32>, %arg16: memref<64x64x3x3xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<128xf32>, %arg21: memref<128x64x3x3xf32>, %arg22: memref<128x128x3x3xf32>, %arg23: memref<128x64x1x1xf32>, %arg24: memref<128xf32>, %arg25: memref<128xf32>, %arg26: memref<128xf32>, %arg27: memref<128xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128x128x3x3xf32>, %arg31: memref<128x128x3x3xf32>, %arg32: memref<256xf32>, %arg33: memref<256xf32>, %arg34: memref<256xf32>, %arg35: memref<256xf32>, %arg36: memref<256x128x3x3xf32>, %arg37: memref<256x256x3x3xf32>, %arg38: memref<256x128x1x1xf32>, %arg39: memref<256xf32>, %arg40: memref<256xf32>, %arg41: memref<256xf32>, %arg42: memref<256xf32>, %arg43: memref<256xf32>, %arg44: memref<256xf32>, %arg45: memref<256x256x3x3xf32>, %arg46: memref<256x256x3x3xf32>, %arg47: memref<512xf32>, %arg48: memref<512xf32>, %arg49: memref<512xf32>, %arg50: memref<512xf32>, %arg51: memref<512x256x3x3xf32>, %arg52: memref<512x512x3x3xf32>, %arg53: memref<512x256x1x1xf32>, %arg54: memref<512xf32>, %arg55: memref<512xf32>, %arg56: memref<512xf32>, %arg57: memref<512xf32>, %arg58: memref<512xf32>, %arg59: memref<512xf32>, %arg60: memref<512x512x3x3xf32>, %arg61: memref<512x512x3x3xf32>, %arg62: memref<i64>, %arg63: memref<64xf32>, %arg64: memref<64xf32>, %arg65: memref<i64>, %arg66: memref<64xf32>, %arg67: memref<64xf32>, %arg68: memref<i64>, %arg69: memref<64xf32>, %arg70: memref<64xf32>, %arg71: memref<i64>, %arg72: memref<64xf32>, %arg73: memref<64xf32>, %arg74: memref<i64>, %arg75: memref<64xf32>, %arg76: memref<64xf32>, %arg77: memref<i64>, %arg78: memref<128xf32>, %arg79: memref<128xf32>, %arg80: memref<i64>, %arg81: memref<128xf32>, %arg82: memref<128xf32>, %arg83: memref<i64>, %arg84: memref<128xf32>, %arg85: memref<128xf32>, %arg86: memref<i64>, %arg87: memref<128xf32>, %arg88: memref<128xf32>, %arg89: memref<i64>, %arg90: memref<128xf32>, %arg91: memref<128xf32>, %arg92: memref<i64>, %arg93: memref<256xf32>, %arg94: memref<256xf32>, %arg95: memref<i64>, %arg96: memref<256xf32>, %arg97: memref<256xf32>, %arg98: memref<i64>, %arg99: memref<256xf32>, %arg100: memref<256xf32>, %arg101: memref<i64>, %arg102: memref<256xf32>, %arg103: memref<256xf32>, %arg104: memref<i64>, %arg105: memref<256xf32>, %arg106: memref<256xf32>, %arg107: memref<i64>, %arg108: memref<512xf32>, %arg109: memref<512xf32>, %arg110: memref<i64>, %arg111: memref<512xf32>, %arg112: memref<512xf32>, %arg113: memref<i64>, %arg114: memref<512xf32>, %arg115: memref<512xf32>, %arg116: memref<i64>, %arg117: memref<512xf32>, %arg118: memref<512xf32>, %arg119: memref<i64>, %arg120: memref<512xf32>, %arg121: memref<512xf32>, %arg122: memref<1x3x224x224xf32>) -> (memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>) {
    %0 = memref.alloc() : memref<f16>
    %1 = memref.alloc() : memref<1x1000xf16>
    %2 = memref.alloc() : memref<512x1000xf16>
    %3 = memref.alloc() : memref<1x512xf16>
    %4 = memref.alloc() : memref<1x512x7x7xf16>
    %5 = memref.alloc() : memref<1x512x7x7xf16>
    %6 = memref.alloc() : memref<1x512x7x7xf16>
    %7 = memref.alloc() : memref<1x512x7x7xf16>
    %8 = memref.alloc() : memref<1x512x7x7xf16>
    %9 = memref.alloc() : memref<1x256x14x14xf16>
    %10 = memref.alloc() : memref<1x256x14x14xf16>
    %11 = memref.alloc() : memref<1x256x14x14xf16>
    %12 = memref.alloc() : memref<1x256x14x14xf16>
    %13 = memref.alloc() : memref<1x256x14x14xf16>
    %14 = memref.alloc() : memref<1x128x28x28xf16>
    %15 = memref.alloc() : memref<1x128x28x28xf16>
    %16 = memref.alloc() : memref<1x128x28x28xf16>
    %17 = memref.alloc() : memref<1x128x28x28xf16>
    %18 = memref.alloc() : memref<1x128x28x28xf16>
    %19 = memref.alloc() : memref<1x64x56x56xf16>
    %20 = memref.alloc() : memref<1x64x56x56xf16>
    %21 = memref.alloc() : memref<1x64x56x56xf16>
    %22 = memref.alloc() : memref<1x64x56x56xf16>
    %23 = memref.alloc() : memref<1x64x56x56xf16>
    %24 = memref.alloc() : memref<1x64x112x112xf16>
    %25 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    "lmhlo.constant"(%25) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %26 = call @Unknown0(%arg122) : (memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16>
    %27 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    lmhlo.convolution(%26, %27, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x3x224x224xf16>, memref<64x3x7x7xf16>, memref<1x64x112x112xf16>) -> ()
    %28:3 = call @BatchNormTrainingOp2(%24, %arg1, %arg0) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %29 = call @Unknown3(%28#0) : (memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    "lmhlo.reduce_window"(%29, %25, %23) ({
    ^bb0(%arg123: memref<f16>, %arg124: memref<f16>, %arg125: memref<f16>):
      %127 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg123, %arg124, %127) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%127, %arg125) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16>, memref<f16>, memref<1x64x56x56xf16>) -> ()
    %30 = call @Unknown4(%arg9) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%23, %30, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %31:3 = call @BatchNormTrainingOp5(%22, %arg6, %arg5) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %32 = call @Unknown6(%31#0) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %33 = call @Unknown7(%arg10) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%32, %33, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %34:3 = call @BatchNormTrainingOp8(%21, %arg8, %arg7) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %35 = call @Unknown9(%34#0, %23) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %36 = call @Unknown10(%arg15) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%35, %36, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %37:3 = call @BatchNormTrainingOp11(%20, %arg12, %arg11) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %38 = call @Unknown12(%37#0) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %39 = call @Unknown13(%arg16) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%38, %39, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %40:3 = call @BatchNormTrainingOp14(%19, %arg14, %arg13) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %41 = call @Unknown15(%40#0, %35) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %42 = call @Unknown16(%arg23) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    lmhlo.convolution(%41, %42, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>) -> ()
    %43:3 = call @BatchNormTrainingOp17(%18, %arg25, %arg24) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %44 = call @Unknown18(%arg21) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    lmhlo.convolution(%41, %44, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %45:3 = call @BatchNormTrainingOp19(%17, %arg18, %arg17) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %46 = call @Unknown20(%45#0) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %47 = call @Unknown21(%arg22) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    lmhlo.convolution(%46, %47, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %48:3 = call @BatchNormTrainingOp22(%16, %arg20, %arg19) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %49 = call @Unknown23(%48#0, %43#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %50 = call @Unknown24(%arg30) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    lmhlo.convolution(%49, %50, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %51:3 = call @BatchNormTrainingOp25(%15, %arg27, %arg26) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %52 = call @Unknown26(%51#0) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %53 = call @Unknown27(%arg31) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    lmhlo.convolution(%52, %53, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %54:3 = call @BatchNormTrainingOp28(%14, %arg29, %arg28) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %55 = call @Unknown29(%54#0, %49) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %56 = call @Unknown30(%arg38) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    lmhlo.convolution(%55, %56, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>) -> ()
    %57:3 = call @BatchNormTrainingOp31(%13, %arg40, %arg39) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %58 = call @Unknown32(%arg36) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    lmhlo.convolution(%55, %58, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %59:3 = call @BatchNormTrainingOp33(%12, %arg33, %arg32) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %60 = call @Unknown34(%59#0) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %61 = call @Unknown35(%arg37) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    lmhlo.convolution(%60, %61, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %62:3 = call @BatchNormTrainingOp36(%11, %arg35, %arg34) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %63 = call @Unknown37(%62#0, %57#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %64 = call @Unknown38(%arg45) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    lmhlo.convolution(%63, %64, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %65:3 = call @BatchNormTrainingOp39(%10, %arg42, %arg41) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %66 = call @Unknown40(%65#0) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %67 = call @Unknown41(%arg46) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    lmhlo.convolution(%66, %67, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %68:3 = call @BatchNormTrainingOp42(%9, %arg44, %arg43) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %69 = call @Unknown43(%68#0, %63) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %70 = call @Unknown44(%arg53) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    lmhlo.convolution(%69, %70, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>) -> ()
    %71:3 = call @BatchNormTrainingOp45(%8, %arg55, %arg54) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %72 = call @Unknown46(%arg51) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    lmhlo.convolution(%69, %72, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %73:3 = call @BatchNormTrainingOp47(%7, %arg48, %arg47) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %74 = call @Unknown48(%73#0) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %75 = call @Unknown49(%arg52) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    lmhlo.convolution(%74, %75, %6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %76:3 = call @BatchNormTrainingOp50(%6, %arg50, %arg49) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %77 = call @Unknown51(%76#0, %71#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %78 = call @Unknown52(%arg60) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    lmhlo.convolution(%77, %78, %5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %79:3 = call @BatchNormTrainingOp53(%5, %arg57, %arg56) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %80 = call @Unknown54(%79#0) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %81 = call @Unknown55(%arg61) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    lmhlo.convolution(%80, %81, %4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %82:3 = call @BatchNormTrainingOp56(%4, %arg59, %arg58) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %83 = call @Unknown57(%82#0, %77) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    "lmhlo.reduce"(%83, %0, %3) ({
    ^bb0(%arg123: memref<f16>, %arg124: memref<f16>, %arg125: memref<f16>):
      "lmhlo.add"(%arg123, %arg124, %arg125) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<1x512x7x7xf16>, memref<f16>, memref<1x512xf16>) -> ()
    %84 = call @Unknown58(%3) : (memref<1x512xf16>) -> memref<1x512xf16>
    %85 = call @Unknown59(%arg4) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    "lmhlo.transpose"(%85, %2) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<1000x512xf16>, memref<512x1000xf16>) -> ()
    "lmhlo.dot"(%84, %85, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512xf16>, memref<1000x512xf16>, memref<1x1000xf16>) -> ()
    %86 = call @Unknown60(%arg3, %1) : (memref<1000xf32>, memref<1x1000xf16>) -> memref<1x1000xf16>
    %87 = call @Unknown61(%28#1, %arg63) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %88 = call @Unknown62(%28#2, %arg64) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %89 = call @Unknown63(%31#1, %arg66) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %90 = call @Unknown64(%31#2, %arg67) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %91 = call @Unknown65(%34#1, %arg69) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %92 = call @Unknown66(%34#2, %arg70) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %93 = call @Unknown67(%37#1, %arg72) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %94 = call @Unknown68(%37#2, %arg73) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %95 = call @Unknown69(%40#1, %arg75) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %96 = call @Unknown70(%40#2, %arg76) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %97 = call @Unknown71(%45#1, %arg78) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %98 = call @Unknown72(%45#2, %arg79) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %99 = call @Unknown73(%48#1, %arg81) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %100 = call @Unknown74(%48#2, %arg82) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %101 = call @Unknown75(%43#1, %arg84) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %102 = call @Unknown76(%43#2, %arg85) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %103 = call @Unknown77(%51#1, %arg87) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %104 = call @Unknown78(%51#2, %arg88) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %105 = call @Unknown79(%54#1, %arg90) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %106 = call @Unknown80(%54#2, %arg91) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %107 = call @Unknown81(%59#1, %arg93) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %108 = call @Unknown82(%59#2, %arg94) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %109 = call @Unknown83(%62#1, %arg96) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %110 = call @Unknown84(%62#2, %arg97) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %111 = call @Unknown85(%57#1, %arg99) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %112 = call @Unknown86(%57#2, %arg100) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %113 = call @Unknown87(%65#1, %arg102) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %114 = call @Unknown88(%65#2, %arg103) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %115 = call @Unknown89(%68#1, %arg105) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %116 = call @Unknown90(%68#2, %arg106) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %117 = call @Unknown91(%73#1, %arg108) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %118 = call @Unknown92(%73#2, %arg109) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %119 = call @Unknown93(%76#1, %arg111) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %120 = call @Unknown94(%76#2, %arg112) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %121 = call @Unknown95(%71#1, %arg114) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %122 = call @Unknown96(%71#2, %arg115) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %123 = call @Unknown97(%79#1, %arg117) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %124 = call @Unknown98(%79#2, %arg118) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %125 = call @Unknown99(%82#1, %arg120) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %126 = call @Unknown100(%82#2, %arg121) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    return %86, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %27, %26, %24, %29, %23, %30, %22, %32, %33, %21, %35, %36, %20, %38, %39, %19, %41, %44, %17, %46, %47, %16, %42, %18, %49, %50, %15, %52, %53, %14, %55, %58, %12, %60, %61, %11, %56, %13, %63, %64, %10, %66, %67, %9, %69, %72, %7, %74, %75, %6, %70, %8, %77, %78, %5, %80, %81, %4, %83, %84, %2 : memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>
  }
}

