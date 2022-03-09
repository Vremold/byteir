// RUN: byteir-opt %s -byre-opt="append-arg-types" | FileCheck %s

// XFAIL: *
module attributes {gpu.container_module} {
  func private @Unknown0(%arg0: memref<32x3x224x224xf32>) -> memref<32x3x224x224xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 150528 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown0_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c150528 = arith.constant 150528 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x3x224x224xf16>
    gpu.launch_func  @Unknown0_kernel::@Unknown0_kernel blocks in (%c150528, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x3x224x224xf32>, %0 : memref<32x3x224x224xf16>)
    return %0 : memref<32x3x224x224xf16>
  }
  gpu.module @Unknown0_kernel {
    gpu.func @Unknown0_kernel(%arg0: memref<32x3x224x224xf32>, %arg1: memref<32x3x224x224xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c4816896 = arith.constant 4816896 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c4816896 : index
      scf.if %5 {
        %c224 = arith.constant 224 : index
        %6 = arith.remsi %4, %c224 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c224 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c224 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c224 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c224 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c224 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c3 = arith.constant 3 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x3x224x224xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x3x224x224xf16>
      }
      gpu.return
    }
  }
  func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 294 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown1_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c294 = arith.constant 294 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x3x7x7xf16>
    gpu.launch_func  @Unknown1_kernel::@Unknown1_kernel blocks in (%c294, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x3x7x7xf32>, %0 : memref<64x3x7x7xf16>)
    return %0 : memref<64x3x7x7xf16>
  }
  gpu.module @Unknown1_kernel {
    gpu.func @Unknown1_kernel(%arg0: memref<64x3x7x7xf32>, %arg1: memref<64x3x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c3 = arith.constant 3 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x3x7x7xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x3x7x7xf16>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp2(%arg0: memref<32x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x112x112xf32>
    %1 = memref.alloc() : memref<32x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x112x112xf32>, memref<32x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown3(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown3_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16000 = arith.constant 16000 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1000x512xf16>
    gpu.launch_func  @Unknown3_kernel::@Unknown3_kernel blocks in (%c16000, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<1000x512xf32>, %0 : memref<1000x512xf16>)
    return %0 : memref<1000x512xf16>
  }
  gpu.module @Unknown3_kernel {
    gpu.func @Unknown3_kernel(%arg0: memref<1000x512xf32>, %arg1: memref<1000x512xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<1000x512xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9] : memref<1000x512xf16>
      }
      gpu.return
    }
  }
  func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown4_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown4_kernel::@Unknown4_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown4_kernel {
    gpu.func @Unknown4_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown5(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown5_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown5_kernel::@Unknown5_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown5_kernel {
    gpu.func @Unknown5_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown6(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown6_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown6_kernel::@Unknown6_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown6_kernel {
    gpu.func @Unknown6_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown7(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown7_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    gpu.launch_func  @Unknown7_kernel::@Unknown7_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf32>, %0 : memref<64x64x3x3xf16>)
    return %0 : memref<64x64x3x3xf16>
  }
  gpu.module @Unknown7_kernel {
    gpu.func @Unknown7_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown8_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2304 = arith.constant 2304 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x64x3x3xf16>
    gpu.launch_func  @Unknown8_kernel::@Unknown8_kernel blocks in (%c2304, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x64x3x3xf32>, %0 : memref<128x64x3x3xf16>)
    return %0 : memref<128x64x3x3xf16>
  }
  gpu.module @Unknown8_kernel {
    gpu.func @Unknown8_kernel(%arg0: memref<128x64x3x3xf32>, %arg1: memref<128x64x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x64x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x64x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown9_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    gpu.launch_func  @Unknown9_kernel::@Unknown9_kernel blocks in (%c4608, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x128x3x3xf32>, %0 : memref<128x128x3x3xf16>)
    return %0 : memref<128x128x3x3xf16>
  }
  gpu.module @Unknown9_kernel {
    gpu.func @Unknown9_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown10(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown10_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x64x1x1xf16>
    gpu.launch_func  @Unknown10_kernel::@Unknown10_kernel blocks in (%c256, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x64x1x1xf32>, %0 : memref<128x64x1x1xf16>)
    return %0 : memref<128x64x1x1xf16>
  }
  gpu.module @Unknown10_kernel {
    gpu.func @Unknown10_kernel(%arg0: memref<128x64x1x1xf32>, %arg1: memref<128x64x1x1xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c64 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<128x64x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<128x64x1x1xf16>
      }
      gpu.return
    }
  }
  func private @Unknown11(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown11_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    gpu.launch_func  @Unknown11_kernel::@Unknown11_kernel blocks in (%c4608, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x128x3x3xf32>, %0 : memref<128x128x3x3xf16>)
    return %0 : memref<128x128x3x3xf16>
  }
  gpu.module @Unknown11_kernel {
    gpu.func @Unknown11_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown12(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown12_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    gpu.launch_func  @Unknown12_kernel::@Unknown12_kernel blocks in (%c4608, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x128x3x3xf32>, %0 : memref<128x128x3x3xf16>)
    return %0 : memref<128x128x3x3xf16>
  }
  gpu.module @Unknown12_kernel {
    gpu.func @Unknown12_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 9216 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown13_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c9216 = arith.constant 9216 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x128x3x3xf16>
    gpu.launch_func  @Unknown13_kernel::@Unknown13_kernel blocks in (%c9216, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x128x3x3xf32>, %0 : memref<256x128x3x3xf16>)
    return %0 : memref<256x128x3x3xf16>
  }
  gpu.module @Unknown13_kernel {
    gpu.func @Unknown13_kernel(%arg0: memref<256x128x3x3xf32>, %arg1: memref<256x128x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x128x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x128x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown14_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    gpu.launch_func  @Unknown14_kernel::@Unknown14_kernel blocks in (%c18432, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x256x3x3xf32>, %0 : memref<256x256x3x3xf16>)
    return %0 : memref<256x256x3x3xf16>
  }
  gpu.module @Unknown14_kernel {
    gpu.func @Unknown14_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown15(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown15_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x128x1x1xf16>
    gpu.launch_func  @Unknown15_kernel::@Unknown15_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x128x1x1xf32>, %0 : memref<256x128x1x1xf16>)
    return %0 : memref<256x128x1x1xf16>
  }
  gpu.module @Unknown15_kernel {
    gpu.func @Unknown15_kernel(%arg0: memref<256x128x1x1xf32>, %arg1: memref<256x128x1x1xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<256x128x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<256x128x1x1xf16>
      }
      gpu.return
    }
  }
  func private @Unknown16(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown16_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    gpu.launch_func  @Unknown16_kernel::@Unknown16_kernel blocks in (%c18432, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x256x3x3xf32>, %0 : memref<256x256x3x3xf16>)
    return %0 : memref<256x256x3x3xf16>
  }
  gpu.module @Unknown16_kernel {
    gpu.func @Unknown16_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown17(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown17_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    gpu.launch_func  @Unknown17_kernel::@Unknown17_kernel blocks in (%c18432, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x256x3x3xf32>, %0 : memref<256x256x3x3xf16>)
    return %0 : memref<256x256x3x3xf16>
  }
  gpu.module @Unknown17_kernel {
    gpu.func @Unknown17_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 36864 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown18_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x256x3x3xf16>
    gpu.launch_func  @Unknown18_kernel::@Unknown18_kernel blocks in (%c36864, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x256x3x3xf32>, %0 : memref<512x256x3x3xf16>)
    return %0 : memref<512x256x3x3xf16>
  }
  gpu.module @Unknown18_kernel {
    gpu.func @Unknown18_kernel(%arg0: memref<512x256x3x3xf32>, %arg1: memref<512x256x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x256x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x256x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 73728 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown19_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    gpu.launch_func  @Unknown19_kernel::@Unknown19_kernel blocks in (%c73728, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x512x3x3xf32>, %0 : memref<512x512x3x3xf16>)
    return %0 : memref<512x512x3x3xf16>
  }
  gpu.module @Unknown19_kernel {
    gpu.func @Unknown19_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown20(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4096 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown20_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x256x1x1xf16>
    gpu.launch_func  @Unknown20_kernel::@Unknown20_kernel blocks in (%c4096, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x256x1x1xf32>, %0 : memref<512x256x1x1xf16>)
    return %0 : memref<512x256x1x1xf16>
  }
  gpu.module @Unknown20_kernel {
    gpu.func @Unknown20_kernel(%arg0: memref<512x256x1x1xf32>, %arg1: memref<512x256x1x1xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c256 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<512x256x1x1xf32>
        %17 = arith.truncf %16 : f32 to f16
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<512x256x1x1xf16>
      }
      gpu.return
    }
  }
  func private @Unknown21(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 73728 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown21_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    gpu.launch_func  @Unknown21_kernel::@Unknown21_kernel blocks in (%c73728, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x512x3x3xf32>, %0 : memref<512x512x3x3xf16>)
    return %0 : memref<512x512x3x3xf16>
  }
  gpu.module @Unknown21_kernel {
    gpu.func @Unknown21_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown22(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 73728 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown22_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    gpu.launch_func  @Unknown22_kernel::@Unknown22_kernel blocks in (%c73728, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x512x3x3xf32>, %0 : memref<512x512x3x3xf16>)
    return %0 : memref<512x512x3x3xf16>
  }
  gpu.module @Unknown22_kernel {
    gpu.func @Unknown22_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
  }
  func private @Unknown23(%arg0: memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<32x64x112x112xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 802816 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown23_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x112x112xi1>
    %1 = memref.alloc() : memref<32x64x112x112xf16>
    gpu.launch_func  @Unknown23_kernel::@Unknown23_kernel blocks in (%c802816, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x112x112xf16>, %1 : memref<32x64x112x112xf16>, %0 : memref<32x64x112x112xi1>)
    return %1, %0 : memref<32x64x112x112xf16>, memref<32x64x112x112xi1>
  }
  gpu.module @Unknown23_kernel {
    gpu.func @Unknown23_kernel(%arg0: memref<32x64x112x112xf16>, %arg1: memref<32x64x112x112xf16>, %arg2: memref<32x64x112x112xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c25690112 = arith.constant 25690112 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c25690112 : index
      scf.if %5 {
        %c112 = arith.constant 112 : index
        %6 = arith.remsi %4, %c112 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c112 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c112 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c112 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c112 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c112 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x112x112xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x64x112x112xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x64x112x112xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp24(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown25(%arg0: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown25_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown25_kernel::@Unknown25_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x56x56xf16>, %1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xi1>)
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  gpu.module @Unknown25_kernel {
    gpu.func @Unknown25_kernel(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x64x56x56xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp26(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown27(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown27_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown27_kernel::@Unknown27_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x56x56xf16>, %arg1 : memref<32x64x56x56xf16>, %1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xi1>)
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  gpu.module @Unknown27_kernel {
    gpu.func @Unknown27_kernel(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xf16>, %arg3: memref<32x64x56x56xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x64x56x56xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp28(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown29(%arg0: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown29_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown29_kernel::@Unknown29_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x56x56xf16>, %1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xi1>)
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  gpu.module @Unknown29_kernel {
    gpu.func @Unknown29_kernel(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x64x56x56xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp30(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown31(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown31_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown31_kernel::@Unknown31_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x56x56xf16>, %arg1 : memref<32x64x56x56xf16>, %1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xi1>)
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  gpu.module @Unknown31_kernel {
    gpu.func @Unknown31_kernel(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xf16>, %arg3: memref<32x64x56x56xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x64x56x56xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp32(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @BatchNormTrainingOp33(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown34(%arg0: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown34_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown34_kernel::@Unknown34_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x128x28x28xf16>, %1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xi1>)
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  gpu.module @Unknown34_kernel {
    gpu.func @Unknown34_kernel(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x128x28x28xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp35(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown36(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown36_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown36_kernel::@Unknown36_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x128x28x28xf16>, %arg1 : memref<32x128x28x28xf16>, %1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xi1>)
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  gpu.module @Unknown36_kernel {
    gpu.func @Unknown36_kernel(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xf16>, %arg3: memref<32x128x28x28xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x128x28x28xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp37(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown38(%arg0: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown38_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown38_kernel::@Unknown38_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x128x28x28xf16>, %1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xi1>)
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  gpu.module @Unknown38_kernel {
    gpu.func @Unknown38_kernel(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x128x28x28xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp39(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown40(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown40_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown40_kernel::@Unknown40_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x128x28x28xf16>, %arg1 : memref<32x128x28x28xf16>, %1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xi1>)
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  gpu.module @Unknown40_kernel {
    gpu.func @Unknown40_kernel(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xf16>, %arg3: memref<32x128x28x28xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x128x28x28xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp41(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @BatchNormTrainingOp42(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown43(%arg0: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown43_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown43_kernel::@Unknown43_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x256x14x14xf16>, %1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xi1>)
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  gpu.module @Unknown43_kernel {
    gpu.func @Unknown43_kernel(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x256x14x14xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp44(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown45(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown45_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown45_kernel::@Unknown45_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x256x14x14xf16>, %arg1 : memref<32x256x14x14xf16>, %1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xi1>)
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  gpu.module @Unknown45_kernel {
    gpu.func @Unknown45_kernel(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xf16>, %arg3: memref<32x256x14x14xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x256x14x14xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp46(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown47(%arg0: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown47_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown47_kernel::@Unknown47_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x256x14x14xf16>, %1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xi1>)
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  gpu.module @Unknown47_kernel {
    gpu.func @Unknown47_kernel(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x256x14x14xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp48(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown49(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown49_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown49_kernel::@Unknown49_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x256x14x14xf16>, %arg1 : memref<32x256x14x14xf16>, %1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xi1>)
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  gpu.module @Unknown49_kernel {
    gpu.func @Unknown49_kernel(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xf16>, %arg3: memref<32x256x14x14xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x256x14x14xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp50(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @BatchNormTrainingOp51(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown52(%arg0: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown52_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c25088 = arith.constant 25088 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512x7x7xi1>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    gpu.launch_func  @Unknown52_kernel::@Unknown52_kernel blocks in (%c25088, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x512x7x7xf16>, %1 : memref<32x512x7x7xf16>, %0 : memref<32x512x7x7xi1>)
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xi1>
  }
  gpu.module @Unknown52_kernel {
    gpu.func @Unknown52_kernel(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x512x7x7xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp53(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown54(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown54_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c25088 = arith.constant 25088 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512x7x7xi1>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    gpu.launch_func  @Unknown54_kernel::@Unknown54_kernel blocks in (%c25088, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x512x7x7xf16>, %arg1 : memref<32x512x7x7xf16>, %1 : memref<32x512x7x7xf16>, %0 : memref<32x512x7x7xi1>)
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xi1>
  }
  gpu.module @Unknown54_kernel {
    gpu.func @Unknown54_kernel(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xf16>, %arg3: memref<32x512x7x7xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %40 = arith.cmpf ogt, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x512x7x7xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp55(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown56(%arg0: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown56_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c25088 = arith.constant 25088 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512x7x7xi1>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    gpu.launch_func  @Unknown56_kernel::@Unknown56_kernel blocks in (%c25088, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x512x7x7xf16>, %1 : memref<32x512x7x7xf16>, %0 : memref<32x512x7x7xi1>)
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xi1>
  }
  gpu.module @Unknown56_kernel {
    gpu.func @Unknown56_kernel(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %38 = arith.cmpf ogt, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x512x7x7xi1>
      }
      gpu.return
    }
  }
  func private @BatchNormTrainingOp57(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown58(%arg0: memref<32x512xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 2 : i32, 4 : i32], __byre__kernel_name = "Unknown58_kernel", __byteir_elementwise_fusion__, arg_offsets = [1 : i32, 2 : i32, 3 : i32, 0 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c25088 = arith.constant 25088 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    gpu.launch_func  @Unknown58_kernel::@Unknown58_kernel blocks in (%c25088, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg1 : memref<32x512x7x7xf16>, %arg2 : memref<32x512x7x7xf16>, %1 : memref<32x512x7x7xf16>, %arg0 : memref<32x512xf16>, %0 : memref<32x512x7x7xf16>)
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xf16>
  }
  gpu.module @Unknown58_kernel {
    gpu.func @Unknown58_kernel(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xf16>, %arg3: memref<32x512xf16>, %arg4: memref<32x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 4.900000e+01 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %40 = memref.load %arg3[%35, %29] : memref<32x512xf16>
        %41 = arith.divf %40, %cst_0 : f16
        %42 = arith.cmpf ogt, %39, %cst : f16
        %43 = select %42, %41, %cst : f16
        memref.store %43, %arg4[%35, %29, %19, %9] : memref<32x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown59(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown59_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown59_kernel::@Unknown59_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown59_kernel {
    gpu.func @Unknown59_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp60(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp61(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp62(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown63(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown63_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c25088 = arith.constant 25088 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    gpu.launch_func  @Unknown63_kernel::@Unknown63_kernel blocks in (%c25088, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x512x7x7xi1>, %arg1 : memref<32x512x7x7xf16>, %0 : memref<32x512x7x7xf16>)
    return %0 : memref<32x512x7x7xf16>
  }
  gpu.module @Unknown63_kernel {
    gpu.func @Unknown63_kernel(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x512x7x7xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown64(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown64_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown64_kernel::@Unknown64_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown64_kernel {
    gpu.func @Unknown64_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp65(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp66(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp67(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown68(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xi1>) -> memref<32x512x7x7xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown68_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 0 : i32, 1 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c25088 = arith.constant 25088 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    gpu.launch_func  @Unknown68_kernel::@Unknown68_kernel blocks in (%c25088, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<32x512x7x7xi1>, %arg0 : memref<32x512x7x7xf16>, %arg1 : memref<32x512x7x7xf16>, %0 : memref<32x512x7x7xf16>)
    return %0 : memref<32x512x7x7xf16>
  }
  gpu.module @Unknown68_kernel {
    gpu.func @Unknown68_kernel(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xf16>, %arg3: memref<32x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x512x7x7xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown69(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown69_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown69_kernel::@Unknown69_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown69_kernel {
    gpu.func @Unknown69_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp70(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp71(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp72(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown73(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 25088 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown73_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c25088 = arith.constant 25088 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    gpu.launch_func  @Unknown73_kernel::@Unknown73_kernel blocks in (%c25088, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x512x7x7xi1>, %arg1 : memref<32x512x7x7xf16>, %0 : memref<32x512x7x7xf16>)
    return %0 : memref<32x512x7x7xf16>
  }
  gpu.module @Unknown73_kernel {
    gpu.func @Unknown73_kernel(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c802816 : index
      scf.if %5 {
        %c7 = arith.constant 7 : index
        %6 = arith.remsi %4, %c7 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c7 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x512x7x7xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x512x7x7xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x512x7x7xf16>
      }
      gpu.return
    }
  }
  func private @Unknown74(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown74_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown74_kernel::@Unknown74_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown74_kernel {
    gpu.func @Unknown74_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp75(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp76(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x256x512xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp77(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func private @Unknown78(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown78_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown78_kernel::@Unknown78_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown78_kernel {
    gpu.func @Unknown78_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c512 = arith.constant 512 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c512 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<512xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<512xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp79(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp80(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<1x1x256x512xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp81(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func private @Unknown82(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown82_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 0 : i32, 1 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown82_kernel::@Unknown82_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<32x256x14x14xi1>, %arg0 : memref<32x256x14x14xf16>, %arg1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xf16>)
    return %0 : memref<32x256x14x14xf16>
  }
  gpu.module @Unknown82_kernel {
    gpu.func @Unknown82_kernel(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xf16>, %arg3: memref<32x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown83(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown83_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown83_kernel::@Unknown83_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown83_kernel {
    gpu.func @Unknown83_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp84(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp85(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp86(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown87(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown87_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown87_kernel::@Unknown87_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x256x14x14xi1>, %arg1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xf16>)
    return %0 : memref<32x256x14x14xf16>
  }
  gpu.module @Unknown87_kernel {
    gpu.func @Unknown87_kernel(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown88(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown88_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown88_kernel::@Unknown88_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown88_kernel {
    gpu.func @Unknown88_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp89(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp90(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp91(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown92(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown92_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 0 : i32, 1 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown92_kernel::@Unknown92_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<32x256x14x14xi1>, %arg0 : memref<32x256x14x14xf16>, %arg1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xf16>)
    return %0 : memref<32x256x14x14xf16>
  }
  gpu.module @Unknown92_kernel {
    gpu.func @Unknown92_kernel(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xf16>, %arg3: memref<32x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown93(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown93_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown93_kernel::@Unknown93_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown93_kernel {
    gpu.func @Unknown93_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp94(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp95(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp96(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown97(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 50176 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown97_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c50176 = arith.constant 50176 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    gpu.launch_func  @Unknown97_kernel::@Unknown97_kernel blocks in (%c50176, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x256x14x14xi1>, %arg1 : memref<32x256x14x14xf16>, %0 : memref<32x256x14x14xf16>)
    return %0 : memref<32x256x14x14xf16>
  }
  gpu.module @Unknown97_kernel {
    gpu.func @Unknown97_kernel(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1605632 = arith.constant 1605632 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1605632 : index
      scf.if %5 {
        %c14 = arith.constant 14 : index
        %6 = arith.remsi %4, %c14 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c14 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c14 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c14 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c14 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c14 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x256x14x14xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x256x14x14xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x256x14x14xf16>
      }
      gpu.return
    }
  }
  func private @Unknown98(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown98_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown98_kernel::@Unknown98_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown98_kernel {
    gpu.func @Unknown98_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp99(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp100(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x128x256xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp101(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func private @Unknown102(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown102_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown102_kernel::@Unknown102_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown102_kernel {
    gpu.func @Unknown102_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp103(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp104(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<1x1x128x256xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp105(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func private @Unknown106(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown106_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 0 : i32, 1 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown106_kernel::@Unknown106_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<32x128x28x28xi1>, %arg0 : memref<32x128x28x28xf16>, %arg1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xf16>)
    return %0 : memref<32x128x28x28xf16>
  }
  gpu.module @Unknown106_kernel {
    gpu.func @Unknown106_kernel(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xf16>, %arg3: memref<32x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown107(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown107_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown107_kernel::@Unknown107_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown107_kernel {
    gpu.func @Unknown107_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp108(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp109(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp110(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown111(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown111_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown111_kernel::@Unknown111_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x128x28x28xi1>, %arg1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xf16>)
    return %0 : memref<32x128x28x28xf16>
  }
  gpu.module @Unknown111_kernel {
    gpu.func @Unknown111_kernel(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown112(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown112_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown112_kernel::@Unknown112_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown112_kernel {
    gpu.func @Unknown112_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp113(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp114(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp115(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown116(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown116_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 0 : i32, 1 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown116_kernel::@Unknown116_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<32x128x28x28xi1>, %arg0 : memref<32x128x28x28xf16>, %arg1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xf16>)
    return %0 : memref<32x128x28x28xf16>
  }
  gpu.module @Unknown116_kernel {
    gpu.func @Unknown116_kernel(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xf16>, %arg3: memref<32x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown117(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown117_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown117_kernel::@Unknown117_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown117_kernel {
    gpu.func @Unknown117_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp118(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp119(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp120(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown121(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 100352 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown121_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c100352 = arith.constant 100352 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    gpu.launch_func  @Unknown121_kernel::@Unknown121_kernel blocks in (%c100352, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x128x28x28xi1>, %arg1 : memref<32x128x28x28xf16>, %0 : memref<32x128x28x28xf16>)
    return %0 : memref<32x128x28x28xf16>
  }
  gpu.module @Unknown121_kernel {
    gpu.func @Unknown121_kernel(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c3211264 = arith.constant 3211264 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c3211264 : index
      scf.if %5 {
        %c28 = arith.constant 28 : index
        %6 = arith.remsi %4, %c28 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c28 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c28 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c28 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c28 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c28 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x128x28x28xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x128x28x28xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x128x28x28xf16>
      }
      gpu.return
    }
  }
  func private @Unknown122(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown122_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown122_kernel::@Unknown122_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown122_kernel {
    gpu.func @Unknown122_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp123(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp124(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x64x128xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp125(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func private @Unknown126(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown126_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown126_kernel::@Unknown126_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown126_kernel {
    gpu.func @Unknown126_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp127(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp128(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<1x1x64x128xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp129(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func private @Unknown130(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown130_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 0 : i32, 1 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown130_kernel::@Unknown130_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<32x64x56x56xi1>, %arg0 : memref<32x64x56x56xf16>, %arg1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xf16>)
    return %0 : memref<32x64x56x56xf16>
  }
  gpu.module @Unknown130_kernel {
    gpu.func @Unknown130_kernel(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xf16>, %arg3: memref<32x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown131(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown131_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown131_kernel::@Unknown131_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown131_kernel {
    gpu.func @Unknown131_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp132(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp133(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp134(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown135(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown135_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown135_kernel::@Unknown135_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x56x56xi1>, %arg1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xf16>)
    return %0 : memref<32x64x56x56xf16>
  }
  gpu.module @Unknown135_kernel {
    gpu.func @Unknown135_kernel(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown136(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown136_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown136_kernel::@Unknown136_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown136_kernel {
    gpu.func @Unknown136_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp137(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp138(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp139(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown140(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown140_kernel", __byteir_elementwise_fusion__, arg_offsets = [2 : i32, 0 : i32, 1 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown140_kernel::@Unknown140_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg2 : memref<32x64x56x56xi1>, %arg0 : memref<32x64x56x56xf16>, %arg1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xf16>)
    return %0 : memref<32x64x56x56xf16>
  }
  gpu.module @Unknown140_kernel {
    gpu.func @Unknown140_kernel(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xf16>, %arg3: memref<32x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = select %36, %39, %cst : f16
        memref.store %40, %arg3[%35, %29, %19, %9] : memref<32x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown141(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown141_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown141_kernel::@Unknown141_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown141_kernel {
    gpu.func @Unknown141_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp142(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp143(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp144(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown145(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown145_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown145_kernel::@Unknown145_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x56x56xi1>, %arg1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xf16>)
    return %0 : memref<32x64x56x56xf16>
  }
  gpu.module @Unknown145_kernel {
    gpu.func @Unknown145_kernel(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown146(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown146_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown146_kernel::@Unknown146_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown146_kernel {
    gpu.func @Unknown146_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp147(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp148(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp149(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown150(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 200704 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown150_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c200704 = arith.constant 200704 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    gpu.launch_func  @Unknown150_kernel::@Unknown150_kernel blocks in (%c200704, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x56x56xf16>, %arg1 : memref<32x64x56x56xf16>, %0 : memref<32x64x56x56xf16>)
    return %0 : memref<32x64x56x56xf16>
  }
  gpu.module @Unknown150_kernel {
    gpu.func @Unknown150_kernel(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c6422528 = arith.constant 6422528 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c6422528 : index
      scf.if %5 {
        %c56 = arith.constant 56 : index
        %6 = arith.remsi %4, %c56 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c56 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c56 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c56 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c56 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c56 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x64x56x56xf16>
      }
      gpu.return
    }
  }
  func private @Unknown151(%arg0: memref<32x64x112x112xi1>, %arg1: memref<32x64x112x112xf16>) -> memref<32x64x112x112xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 802816 : i32, __byre__arg_ranks = [4 : i32, 4 : i32, 4 : i32], __byre__kernel_name = "Unknown151_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c802816 = arith.constant 802816 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x64x112x112xf16>
    gpu.launch_func  @Unknown151_kernel::@Unknown151_kernel blocks in (%c802816, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x64x112x112xi1>, %arg1 : memref<32x64x112x112xf16>, %0 : memref<32x64x112x112xf16>)
    return %0 : memref<32x64x112x112xf16>
  }
  gpu.module @Unknown151_kernel {
    gpu.func @Unknown151_kernel(%arg0: memref<32x64x112x112xi1>, %arg1: memref<32x64x112x112xf16>, %arg2: memref<32x64x112x112xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c25690112 = arith.constant 25690112 : index
      %cst = arith.constant 0.000000e+00 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c25690112 : index
      scf.if %5 {
        %c112 = arith.constant 112 : index
        %6 = arith.remsi %4, %c112 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c112 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c112 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c112 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c112 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c112 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<32x64x112x112xi1>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<32x64x112x112xf16>
        %38 = select %36, %37, %cst : f16
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<32x64x112x112xf16>
      }
      gpu.return
    }
  }
  func private @Unknown152(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown152_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown152_kernel::@Unknown152_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown152_kernel {
    gpu.func @Unknown152_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = arith.addf %6, %cst : f32
        %8 = math.rsqrt %7 : f32
        %9 = arith.divf %cst_0, %8 : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.subf %10, %cst : f32
        memref.store %11, %arg1[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @BatchNormGradOp153(%arg0: memref<32x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x112x112xf32>
    %1 = memref.alloc() : memref<32x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x112x112xf32>
    %5 = memref.alloc() : memref<32x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x112x112xf32>, memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x112x112xf32>, memref<32x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardFilterOp154(%arg0: memref<32x3x224x224xf16>, %arg1: memref<32x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x3x224x224xf16>, memref<32x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func private @Unknown155(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 294 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown155_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c294 = arith.constant 294 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x3x7x7xf32>
    gpu.launch_func  @Unknown155_kernel::@Unknown155_kernel blocks in (%c294, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x3x7x7xf16>, %0 : memref<64x3x7x7xf32>)
    return %0 : memref<64x3x7x7xf32>
  }
  gpu.module @Unknown155_kernel {
    gpu.func @Unknown155_kernel(%arg0: memref<64x3x7x7xf16>, %arg1: memref<64x3x7x7xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c7 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c7 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c7 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c7 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c3 = arith.constant 3 : index
        %26 = arith.remsi %25, %c3 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c3 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c3 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x3x7x7xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x3x7x7xf32>
      }
      gpu.return
    }
  }
  func private @Unknown156(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown156_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @Unknown156_kernel::@Unknown156_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %0 : memref<64x64x3x3xf32>)
    return %0 : memref<64x64x3x3xf32>
  }
  gpu.module @Unknown156_kernel {
    gpu.func @Unknown156_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown157(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown157_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @Unknown157_kernel::@Unknown157_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %0 : memref<64x64x3x3xf32>)
    return %0 : memref<64x64x3x3xf32>
  }
  gpu.module @Unknown157_kernel {
    gpu.func @Unknown157_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown158(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown158_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @Unknown158_kernel::@Unknown158_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %0 : memref<64x64x3x3xf32>)
    return %0 : memref<64x64x3x3xf32>
  }
  gpu.module @Unknown158_kernel {
    gpu.func @Unknown158_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown159(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1152 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown159_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    gpu.launch_func  @Unknown159_kernel::@Unknown159_kernel blocks in (%c1152, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64x64x3x3xf16>, %0 : memref<64x64x3x3xf32>)
    return %0 : memref<64x64x3x3xf32>
  }
  gpu.module @Unknown159_kernel {
    gpu.func @Unknown159_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<64x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown160(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2304 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown160_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2304 = arith.constant 2304 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x64x3x3xf32>
    gpu.launch_func  @Unknown160_kernel::@Unknown160_kernel blocks in (%c2304, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x64x3x3xf16>, %0 : memref<128x64x3x3xf32>)
    return %0 : memref<128x64x3x3xf32>
  }
  gpu.module @Unknown160_kernel {
    gpu.func @Unknown160_kernel(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c64 = arith.constant 64 : index
        %26 = arith.remsi %25, %c64 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c64 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c64 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x64x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x64x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown161(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown161_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    gpu.launch_func  @Unknown161_kernel::@Unknown161_kernel blocks in (%c4608, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x128x3x3xf16>, %0 : memref<128x128x3x3xf32>)
    return %0 : memref<128x128x3x3xf32>
  }
  gpu.module @Unknown161_kernel {
    gpu.func @Unknown161_kernel(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown162(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown162_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x64x1x1xf32>
    gpu.launch_func  @Unknown162_kernel::@Unknown162_kernel blocks in (%c256, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x64x1x1xf16>, %0 : memref<128x64x1x1xf32>)
    return %0 : memref<128x64x1x1xf32>
  }
  gpu.module @Unknown162_kernel {
    gpu.func @Unknown162_kernel(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c64 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<128x64x1x1xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<128x64x1x1xf32>
      }
      gpu.return
    }
  }
  func private @Unknown163(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown163_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    gpu.launch_func  @Unknown163_kernel::@Unknown163_kernel blocks in (%c4608, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x128x3x3xf16>, %0 : memref<128x128x3x3xf32>)
    return %0 : memref<128x128x3x3xf32>
  }
  gpu.module @Unknown163_kernel {
    gpu.func @Unknown163_kernel(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown164(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4608 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown164_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4608 = arith.constant 4608 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    gpu.launch_func  @Unknown164_kernel::@Unknown164_kernel blocks in (%c4608, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x128x3x3xf16>, %0 : memref<128x128x3x3xf32>)
    return %0 : memref<128x128x3x3xf32>
  }
  gpu.module @Unknown164_kernel {
    gpu.func @Unknown164_kernel(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<128x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown165(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 9216 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown165_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c9216 = arith.constant 9216 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x128x3x3xf32>
    gpu.launch_func  @Unknown165_kernel::@Unknown165_kernel blocks in (%c9216, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x128x3x3xf16>, %0 : memref<256x128x3x3xf32>)
    return %0 : memref<256x128x3x3xf32>
  }
  gpu.module @Unknown165_kernel {
    gpu.func @Unknown165_kernel(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c128 = arith.constant 128 : index
        %26 = arith.remsi %25, %c128 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c128 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c128 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x128x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x128x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown166(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown166_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    gpu.launch_func  @Unknown166_kernel::@Unknown166_kernel blocks in (%c18432, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x256x3x3xf16>, %0 : memref<256x256x3x3xf32>)
    return %0 : memref<256x256x3x3xf32>
  }
  gpu.module @Unknown166_kernel {
    gpu.func @Unknown166_kernel(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown167(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1024 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown167_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x128x1x1xf32>
    gpu.launch_func  @Unknown167_kernel::@Unknown167_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x128x1x1xf16>, %0 : memref<256x128x1x1xf32>)
    return %0 : memref<256x128x1x1xf32>
  }
  gpu.module @Unknown167_kernel {
    gpu.func @Unknown167_kernel(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<256x128x1x1xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<256x128x1x1xf32>
      }
      gpu.return
    }
  }
  func private @Unknown168(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown168_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    gpu.launch_func  @Unknown168_kernel::@Unknown168_kernel blocks in (%c18432, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x256x3x3xf16>, %0 : memref<256x256x3x3xf32>)
    return %0 : memref<256x256x3x3xf32>
  }
  gpu.module @Unknown168_kernel {
    gpu.func @Unknown168_kernel(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown169(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 18432 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown169_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c18432 = arith.constant 18432 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    gpu.launch_func  @Unknown169_kernel::@Unknown169_kernel blocks in (%c18432, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256x256x3x3xf16>, %0 : memref<256x256x3x3xf32>)
    return %0 : memref<256x256x3x3xf32>
  }
  gpu.module @Unknown169_kernel {
    gpu.func @Unknown169_kernel(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<256x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown170(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 36864 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown170_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c36864 = arith.constant 36864 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x256x3x3xf32>
    gpu.launch_func  @Unknown170_kernel::@Unknown170_kernel blocks in (%c36864, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x256x3x3xf16>, %0 : memref<512x256x3x3xf32>)
    return %0 : memref<512x256x3x3xf32>
  }
  gpu.module @Unknown170_kernel {
    gpu.func @Unknown170_kernel(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c256 = arith.constant 256 : index
        %26 = arith.remsi %25, %c256 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c256 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c256 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x256x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x256x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown171(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 73728 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown171_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    gpu.launch_func  @Unknown171_kernel::@Unknown171_kernel blocks in (%c73728, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x512x3x3xf16>, %0 : memref<512x512x3x3xf32>)
    return %0 : memref<512x512x3x3xf32>
  }
  gpu.module @Unknown171_kernel {
    gpu.func @Unknown171_kernel(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown172(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4096 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown172_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x256x1x1xf32>
    gpu.launch_func  @Unknown172_kernel::@Unknown172_kernel blocks in (%c4096, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x256x1x1xf16>, %0 : memref<512x256x1x1xf32>)
    return %0 : memref<512x256x1x1xf32>
  }
  gpu.module @Unknown172_kernel {
    gpu.func @Unknown172_kernel(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c256 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9, %c0, %c0] : memref<512x256x1x1xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9, %c0, %c0] : memref<512x256x1x1xf32>
      }
      gpu.return
    }
  }
  func private @Unknown173(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 73728 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown173_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    gpu.launch_func  @Unknown173_kernel::@Unknown173_kernel blocks in (%c73728, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x512x3x3xf16>, %0 : memref<512x512x3x3xf32>)
    return %0 : memref<512x512x3x3xf32>
  }
  gpu.module @Unknown173_kernel {
    gpu.func @Unknown173_kernel(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown174(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 73728 : i32, __byre__arg_ranks = [4 : i32, 4 : i32], __byre__kernel_name = "Unknown174_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c73728 = arith.constant 73728 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    gpu.launch_func  @Unknown174_kernel::@Unknown174_kernel blocks in (%c73728, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512x512x3x3xf16>, %0 : memref<512x512x3x3xf32>)
    return %0 : memref<512x512x3x3xf32>
  }
  gpu.module @Unknown174_kernel {
    gpu.func @Unknown174_kernel(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c3 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c3 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c3 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c3 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %c512 = arith.constant 512 : index
        %26 = arith.remsi %25, %c512 : index
        %27 = arith.cmpi slt, %26, %c0 : index
        %28 = arith.addi %26, %c512 : index
        %29 = select %27, %28, %26 : index
        %30 = arith.cmpi slt, %25, %c0 : index
        %31 = arith.subi %c-1, %25 : index
        %32 = select %30, %31, %25 : index
        %33 = arith.divsi %32, %c512 : index
        %34 = arith.subi %c-1, %33 : index
        %35 = select %30, %34, %33 : index
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<512x512x3x3xf16>
        %37 = arith.extf %36 : f16 to f32
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
  }
  func private @Unknown175(%arg0: memref<32x512xf16>) -> memref<32x512xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 512 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown175_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x512xf16>
    gpu.launch_func  @Unknown175_kernel::@Unknown175_kernel blocks in (%c512, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x512xf16>, %0 : memref<32x512xf16>)
    return %0 : memref<32x512xf16>
  }
  gpu.module @Unknown175_kernel {
    gpu.func @Unknown175_kernel(%arg0: memref<32x512xf16>, %arg1: memref<32x512xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c16384 = arith.constant 16384 : index
      %cst = arith.constant 2.040100e-02 : f16
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c16384 : index
      scf.if %5 {
        %c512 = arith.constant 512 : index
        %6 = arith.remsi %4, %c512 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c512 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<32x512xf16>
        %17 = arith.mulf %16, %cst : f16
        memref.store %17, %arg1[%15, %9] : memref<32x512xf16>
      }
      gpu.return
    }
  }
  func private @MatmulOp176(%arg0: memref<32x512xf16>, %arg1: memref<32x1000xf16>) -> memref<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<512x1000xf16>
    %1 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512xf16>, memref<32x1000xf16>, memref<512x1000xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<512x1000xf16>, memref<1000x512xf16>) -> ()
    return %1 : memref<1000x512xf16>
  }
  func private @Unknown177(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown177_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16000 = arith.constant 16000 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1000x512xf32>
    gpu.launch_func  @Unknown177_kernel::@Unknown177_kernel blocks in (%c16000, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<1000x512xf16>, %0 : memref<1000x512xf32>)
    return %0 : memref<1000x512xf32>
  }
  gpu.module @Unknown177_kernel {
    gpu.func @Unknown177_kernel(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<1000x512xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9] : memref<1000x512xf32>
      }
      gpu.return
    }
  }
  func private @Unknown178(%arg0: memref<32x1000xf16>) -> memref<32x1000xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1000 : i32, __byre__arg_ranks = [2 : i32, 2 : i32], __byre__kernel_name = "Unknown178_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x1000xf32>
    gpu.launch_func  @Unknown178_kernel::@Unknown178_kernel blocks in (%c1000, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<32x1000xf16>, %0 : memref<32x1000xf32>)
    return %0 : memref<32x1000xf32>
  }
  gpu.module @Unknown178_kernel {
    gpu.func @Unknown178_kernel(%arg0: memref<32x1000xf16>, %arg1: memref<32x1000xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32000 = arith.constant 32000 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32000 : index
      scf.if %5 {
        %c1000 = arith.constant 1000 : index
        %6 = arith.remsi %4, %c1000 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c1000 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c1000 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<32x1000xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9] : memref<32x1000xf32>
      }
      gpu.return
    }
  }
  func private @Unknown179(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown179_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1000xf32>
    gpu.launch_func  @Unknown179_kernel::@Unknown179_kernel blocks in (%c32, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<1000xf32>, %0 : memref<1000xf32>)
    return %0 : memref<1000xf32>
  }
  gpu.module @Unknown179_kernel {
    gpu.func @Unknown179_kernel(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1000 = arith.constant 1000 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1000 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<1000xf32>
        %7 = arith.truncf %6 : f32 to f16
        %8 = arith.extf %7 : f16 to f32
        memref.store %8, %arg1[%4] : memref<1000xf32>
      }
      gpu.return
    }
  }
  func private @Unknown180(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown180_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown180_kernel::@Unknown180_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown180_kernel {
    gpu.func @Unknown180_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown181(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown181_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown181_kernel::@Unknown181_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown181_kernel {
    gpu.func @Unknown181_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown182(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown182_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown182_kernel::@Unknown182_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown182_kernel {
    gpu.func @Unknown182_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown183(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown183_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown183_kernel::@Unknown183_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown183_kernel {
    gpu.func @Unknown183_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown184(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown184_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown184_kernel::@Unknown184_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown184_kernel {
    gpu.func @Unknown184_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown185(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown185_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown185_kernel::@Unknown185_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown185_kernel {
    gpu.func @Unknown185_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown186(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown186_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown186_kernel::@Unknown186_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown186_kernel {
    gpu.func @Unknown186_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown187(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown187_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown187_kernel::@Unknown187_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown187_kernel {
    gpu.func @Unknown187_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown188(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown188_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown188_kernel::@Unknown188_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown188_kernel {
    gpu.func @Unknown188_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown189(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown189_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64xf32>
    gpu.launch_func  @Unknown189_kernel::@Unknown189_kernel blocks in (%c2, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<64xf32>, %arg1 : memref<64xf32>, %0 : memref<64xf32>)
    return %0 : memref<64xf32>
  }
  gpu.module @Unknown189_kernel {
    gpu.func @Unknown189_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c64 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<64xf32>
        %7 = memref.load %arg1[%4] : memref<64xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<64xf32>
      }
      gpu.return
    }
  }
  func private @Unknown190(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown190_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown190_kernel::@Unknown190_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown190_kernel {
    gpu.func @Unknown190_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown191(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown191_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown191_kernel::@Unknown191_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown191_kernel {
    gpu.func @Unknown191_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown192(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown192_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown192_kernel::@Unknown192_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown192_kernel {
    gpu.func @Unknown192_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown193(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown193_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown193_kernel::@Unknown193_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown193_kernel {
    gpu.func @Unknown193_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown194(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown194_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown194_kernel::@Unknown194_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown194_kernel {
    gpu.func @Unknown194_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown195(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown195_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown195_kernel::@Unknown195_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown195_kernel {
    gpu.func @Unknown195_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown196(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown196_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown196_kernel::@Unknown196_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown196_kernel {
    gpu.func @Unknown196_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown197(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown197_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown197_kernel::@Unknown197_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown197_kernel {
    gpu.func @Unknown197_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown198(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown198_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown198_kernel::@Unknown198_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown198_kernel {
    gpu.func @Unknown198_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown199(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 4 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown199_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<128xf32>
    gpu.launch_func  @Unknown199_kernel::@Unknown199_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>, %0 : memref<128xf32>)
    return %0 : memref<128xf32>
  }
  gpu.module @Unknown199_kernel {
    gpu.func @Unknown199_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 1.000000e-01 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c128 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<128xf32>
        %7 = memref.load %arg1[%4] : memref<128xf32>
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.mulf %6, %cst_0 : f32
        %10 = arith.addf %9, %8 : f32
        memref.store %10, %arg2[%4] : memref<128xf32>
      }
      gpu.return
    }
  }
  func private @Unknown200(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown200_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown200_kernel::@Unknown200_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown200_kernel {
    gpu.func @Unknown200_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown201(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown201_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown201_kernel::@Unknown201_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown201_kernel {
    gpu.func @Unknown201_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown202(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown202_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown202_kernel::@Unknown202_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown202_kernel {
    gpu.func @Unknown202_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown203(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown203_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown203_kernel::@Unknown203_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown203_kernel {
    gpu.func @Unknown203_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown204(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown204_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown204_kernel::@Unknown204_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown204_kernel {
    gpu.func @Unknown204_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown205(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown205_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown205_kernel::@Unknown205_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown205_kernel {
    gpu.func @Unknown205_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown206(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown206_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown206_kernel::@Unknown206_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown206_kernel {
    gpu.func @Unknown206_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown207(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown207_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown207_kernel::@Unknown207_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown207_kernel {
    gpu.func @Unknown207_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown208(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown208_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown208_kernel::@Unknown208_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown208_kernel {
    gpu.func @Unknown208_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown209(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 8 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown209_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<256xf32>
    gpu.launch_func  @Unknown209_kernel::@Unknown209_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  gpu.module @Unknown209_kernel {
    gpu.func @Unknown209_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown210(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown210_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown210_kernel::@Unknown210_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown210_kernel {
    gpu.func @Unknown210_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown211(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown211_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown211_kernel::@Unknown211_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown211_kernel {
    gpu.func @Unknown211_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown212(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown212_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown212_kernel::@Unknown212_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown212_kernel {
    gpu.func @Unknown212_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown213(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown213_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown213_kernel::@Unknown213_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown213_kernel {
    gpu.func @Unknown213_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown214(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown214_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown214_kernel::@Unknown214_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown214_kernel {
    gpu.func @Unknown214_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown215(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown215_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown215_kernel::@Unknown215_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown215_kernel {
    gpu.func @Unknown215_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown216(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown216_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown216_kernel::@Unknown216_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown216_kernel {
    gpu.func @Unknown216_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown217(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown217_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown217_kernel::@Unknown217_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown217_kernel {
    gpu.func @Unknown217_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown218(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown218_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown218_kernel::@Unknown218_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown218_kernel {
    gpu.func @Unknown218_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown219(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 16 : i32, __byre__arg_ranks = [1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown219_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<512xf32>
    gpu.launch_func  @Unknown219_kernel::@Unknown219_kernel blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>, %0 : memref<512xf32>)
    return %0 : memref<512xf32>
  }
  gpu.module @Unknown219_kernel {
    gpu.func @Unknown219_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
  func private @Unknown220(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 32 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown220_kernel", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1000xf16>
    gpu.launch_func  @Unknown220_kernel::@Unknown220_kernel blocks in (%c32, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<1000xf32>, %0 : memref<1000xf16>)
    return %0 : memref<1000xf16>
  }
  gpu.module @Unknown220_kernel {
    gpu.func @Unknown220_kernel(%arg0: memref<1000xf32>, %arg1: memref<1000xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1000 = arith.constant 1000 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c1000 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<1000xf32>
        %7 = arith.truncf %6 : f32 to f16
        memref.store %7, %arg1[%4] : memref<1000xf16>
      }
      gpu.return
    }
  }
  func private @Unknown221(%arg0: memref<1000xf16>, %arg1: memref<32x1000xf16>) -> memref<32x1000xf16> attributes {__byre__BlockSize.x = 32 : i32, __byre__GridSize.x = 1000 : i32, __byre__arg_ranks = [2 : i32, 1 : i32, 2 : i32], __byre__kernel_name = "Unknown221_kernel", __byteir_elementwise_fusion__, arg_offsets = [1 : i32, 0 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<32x1000xf16>
    gpu.launch_func  @Unknown221_kernel::@Unknown221_kernel blocks in (%c1000, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg1 : memref<32x1000xf16>, %arg0 : memref<1000xf16>, %0 : memref<32x1000xf16>)
    return %0 : memref<32x1000xf16>
  }
  gpu.module @Unknown221_kernel {
    gpu.func @Unknown221_kernel(%arg0: memref<32x1000xf16>, %arg1: memref<1000xf16>, %arg2: memref<32x1000xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32000 = arith.constant 32000 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32000 : index
      scf.if %5 {
        %c1000 = arith.constant 1000 : index
        %6 = arith.remsi %4, %c1000 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c1000 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c1000 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<32x1000xf16>
        %17 = memref.load %arg1[%9] : memref<1000xf16>
        %18 = arith.addf %16, %17 : f16
        memref.store %18, %arg2[%15, %9] : memref<32x1000xf16>
      }
      gpu.return
    }
  }
  func @main(%arg0: memref<64x3x7x7xf32>, %arg1: memref<32x3x224x224xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64x64x3x3xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<64xf32>, %arg11: memref<64x64x3x3xf32>, %arg12: memref<64xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64xf32>, %arg16: memref<64x64x3x3xf32>, %arg17: memref<64xf32>, %arg18: memref<64xf32>, %arg19: memref<64xf32>, %arg20: memref<64xf32>, %arg21: memref<64x64x3x3xf32>, %arg22: memref<64xf32>, %arg23: memref<64xf32>, %arg24: memref<64xf32>, %arg25: memref<64xf32>, %arg26: memref<128x64x3x3xf32>, %arg27: memref<128xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128x3x3xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<128xf32>, %arg37: memref<128xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128x64x1x1xf32>, %arg41: memref<128x128x3x3xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<128x128x3x3xf32>, %arg47: memref<128xf32>, %arg48: memref<128xf32>, %arg49: memref<128xf32>, %arg50: memref<128xf32>, %arg51: memref<256x128x3x3xf32>, %arg52: memref<256xf32>, %arg53: memref<256xf32>, %arg54: memref<256xf32>, %arg55: memref<256xf32>, %arg56: memref<256x256x3x3xf32>, %arg57: memref<256xf32>, %arg58: memref<256xf32>, %arg59: memref<256xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256x128x1x1xf32>, %arg66: memref<256x256x3x3xf32>, %arg67: memref<256xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<256xf32>, %arg71: memref<256x256x3x3xf32>, %arg72: memref<256xf32>, %arg73: memref<256xf32>, %arg74: memref<256xf32>, %arg75: memref<256xf32>, %arg76: memref<512x256x3x3xf32>, %arg77: memref<512xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<512xf32>, %arg81: memref<512x512x3x3xf32>, %arg82: memref<512xf32>, %arg83: memref<512xf32>, %arg84: memref<512xf32>, %arg85: memref<512xf32>, %arg86: memref<512xf32>, %arg87: memref<512xf32>, %arg88: memref<512xf32>, %arg89: memref<512xf32>, %arg90: memref<512x256x1x1xf32>, %arg91: memref<512x512x3x3xf32>, %arg92: memref<512xf32>, %arg93: memref<512xf32>, %arg94: memref<512xf32>, %arg95: memref<512xf32>, %arg96: memref<512x512x3x3xf32>, %arg97: memref<512xf32>, %arg98: memref<512xf32>, %arg99: memref<512xf32>, %arg100: memref<512xf32>, %arg101: memref<1000x512xf32>, %arg102: memref<32x1000xf16>, %arg103: memref<1000xf32>) -> (memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<32x1000xf16>) {
    %0 = memref.alloc() : memref<i64>
    %1 = memref.alloc() : memref<32x1000xf16>
    %2 = memref.alloc() : memref<1000xf32>
    %3 = memref.alloc() : memref<32x512xf16>
    %4 = memref.alloc() : memref<32x64x112x112xf16>
    %5 = memref.alloc() : memref<32x512x7x7xf16>
    %6 = memref.alloc() : memref<32x512x7x7xf16>
    %7 = memref.alloc() : memref<32x512x7x7xf16>
    %8 = memref.alloc() : memref<32x512x7x7xf16>
    %9 = memref.alloc() : memref<32x512x7x7xf16>
    %10 = memref.alloc() : memref<32x256x14x14xf16>
    %11 = memref.alloc() : memref<32x256x14x14xf16>
    %12 = memref.alloc() : memref<32x256x14x14xf16>
    %13 = memref.alloc() : memref<32x256x14x14xf16>
    %14 = memref.alloc() : memref<32x256x14x14xf16>
    %15 = memref.alloc() : memref<32x128x28x28xf16>
    %16 = memref.alloc() : memref<32x128x28x28xf16>
    %17 = memref.alloc() : memref<32x128x28x28xf16>
    %18 = memref.alloc() : memref<32x128x28x28xf16>
    %19 = memref.alloc() : memref<32x128x28x28xf16>
    %20 = memref.alloc() : memref<32x64x56x56xf16>
    %21 = memref.alloc() : memref<32x64x56x56xf16>
    %22 = memref.alloc() : memref<32x64x56x56xf16>
    %23 = memref.alloc() : memref<32x64x56x56xf16>
    %24 = memref.alloc() : memref<32x64x56x56xf16>
    %25 = memref.alloc() : memref<32x512xf16>
    %26 = memref.alloc() : memref<32x64x112x112xf16>
    %27 = memref.alloc() : memref<f16>
    %28 = memref.alloc() : memref<f16>
    %29 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<1> : tensor<i64>} : (memref<i64>) -> ()
    "lmhlo.constant"(%29) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%28) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    "lmhlo.constant"(%27) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %30 = call @Unknown0(%arg1) : (memref<32x3x224x224xf32>) -> memref<32x3x224x224xf16>
    %31 = call @Unknown1(%arg0) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    lmhlo.convolution(%30, %31, %26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x3x224x224xf16>, memref<64x3x7x7xf16>, memref<32x64x112x112xf16>) -> ()
    %32:3 = call @BatchNormTrainingOp2(%26, %arg5, %arg4) : (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %33 = call @Unknown3(%arg101) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    "lmhlo.dot"(%arg102, %33, %25) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x1000xf16>, memref<1000x512xf16>, memref<32x512xf16>) -> ()
    %34 = call @Unknown4(%arg6) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %35 = call @Unknown5(%arg11) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %36 = call @Unknown6(%arg16) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %37 = call @Unknown7(%arg21) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %38 = call @Unknown8(%arg26) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %39 = call @Unknown9(%arg31) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %40 = call @Unknown10(%arg40) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %41 = call @Unknown11(%arg41) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %42 = call @Unknown12(%arg46) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %43 = call @Unknown13(%arg51) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %44 = call @Unknown14(%arg56) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %45 = call @Unknown15(%arg65) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %46 = call @Unknown16(%arg66) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %47 = call @Unknown17(%arg71) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %48 = call @Unknown18(%arg76) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %49 = call @Unknown19(%arg81) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %50 = call @Unknown20(%arg90) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %51 = call @Unknown21(%arg91) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %52 = call @Unknown22(%arg96) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %53:2 = call @Unknown23(%32#0) : (memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<32x64x112x112xi1>)
    "lmhlo.reduce_window"(%53#0, %27, %24) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      %252 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg104, %arg105, %252) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%252, %arg106) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<32x64x112x112xf16>, memref<f16>, memref<32x64x56x56xf16>) -> ()
    lmhlo.convolution(%24, %34, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %54:3 = call @BatchNormTrainingOp24(%23, %arg10, %arg9) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %55:2 = call @Unknown25(%54#0) : (memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%55#0, %35, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %56:3 = call @BatchNormTrainingOp26(%22, %arg15, %arg14) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %57:2 = call @Unknown27(%56#0, %24) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%57#0, %36, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %58:3 = call @BatchNormTrainingOp28(%21, %arg20, %arg19) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %59:2 = call @Unknown29(%58#0) : (memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%59#0, %37, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %60:3 = call @BatchNormTrainingOp30(%20, %arg25, %arg24) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %61:2 = call @Unknown31(%60#0, %57#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%61#0, %38, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<128x64x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %62:3 = call @BatchNormTrainingOp32(%19, %arg30, %arg29) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    lmhlo.convolution(%61#0, %40, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<128x64x1x1xf16>, memref<32x128x28x28xf16>) -> ()
    %63:3 = call @BatchNormTrainingOp33(%18, %arg39, %arg38) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %64:2 = call @Unknown34(%62#0) : (memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%64#0, %39, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %65:3 = call @BatchNormTrainingOp35(%17, %arg35, %arg34) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %66:2 = call @Unknown36(%65#0, %63#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%66#0, %41, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %67:3 = call @BatchNormTrainingOp37(%16, %arg45, %arg44) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %68:2 = call @Unknown38(%67#0) : (memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%68#0, %42, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %69:3 = call @BatchNormTrainingOp39(%15, %arg50, %arg49) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %70:2 = call @Unknown40(%69#0, %66#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%70#0, %43, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<256x128x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %71:3 = call @BatchNormTrainingOp41(%14, %arg55, %arg54) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    lmhlo.convolution(%70#0, %45, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<256x128x1x1xf16>, memref<32x256x14x14xf16>) -> ()
    %72:3 = call @BatchNormTrainingOp42(%13, %arg64, %arg63) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %73:2 = call @Unknown43(%71#0) : (memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%73#0, %44, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %74:3 = call @BatchNormTrainingOp44(%12, %arg60, %arg59) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %75:2 = call @Unknown45(%74#0, %72#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%75#0, %46, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %76:3 = call @BatchNormTrainingOp46(%11, %arg70, %arg69) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %77:2 = call @Unknown47(%76#0) : (memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%77#0, %47, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %78:3 = call @BatchNormTrainingOp48(%10, %arg75, %arg74) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %79:2 = call @Unknown49(%78#0, %75#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%79#0, %48, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<512x256x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %80:3 = call @BatchNormTrainingOp50(%9, %arg80, %arg79) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    lmhlo.convolution(%79#0, %50, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<512x256x1x1xf16>, memref<32x512x7x7xf16>) -> ()
    %81:3 = call @BatchNormTrainingOp51(%8, %arg89, %arg88) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %82:2 = call @Unknown52(%80#0) : (memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>)
    lmhlo.convolution(%82#0, %49, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %83:3 = call @BatchNormTrainingOp53(%7, %arg85, %arg84) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %84:2 = call @Unknown54(%83#0, %81#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>)
    lmhlo.convolution(%84#0, %51, %6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %85:3 = call @BatchNormTrainingOp55(%6, %arg95, %arg94) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %86:2 = call @Unknown56(%85#0) : (memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>)
    lmhlo.convolution(%86#0, %52, %5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %87:3 = call @BatchNormTrainingOp57(%5, %arg100, %arg99) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %88:2 = call @Unknown58(%25, %87#0, %84#0) : (memref<32x512xf16>, memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>)
    %89 = call @Unknown59(%87#2) : (memref<512xf32>) -> memref<512xf32>
    %90:3 = call @BatchNormGradOp60(%5, %arg100, %87#1, %89, %88#1) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %91 = call @ConvBackwardDataOp61(%90#0, %52) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %92 = call @ConvBackwardFilterOp62(%86#0, %90#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %93 = call @Unknown63(%86#1, %91) : (memref<32x512x7x7xi1>, memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16>
    %94 = call @Unknown64(%85#2) : (memref<512xf32>) -> memref<512xf32>
    %95:3 = call @BatchNormGradOp65(%6, %arg95, %85#1, %94, %93) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %96 = call @ConvBackwardDataOp66(%95#0, %51) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %97 = call @ConvBackwardFilterOp67(%84#0, %95#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %98 = call @Unknown68(%88#1, %96, %84#1) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) -> memref<32x512x7x7xf16>
    %99 = call @Unknown69(%83#2) : (memref<512xf32>) -> memref<512xf32>
    %100:3 = call @BatchNormGradOp70(%7, %arg85, %83#1, %99, %98) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %101 = call @ConvBackwardDataOp71(%100#0, %49) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %102 = call @ConvBackwardFilterOp72(%82#0, %100#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %103 = call @Unknown73(%82#1, %101) : (memref<32x512x7x7xi1>, memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16>
    %104 = call @Unknown74(%80#2) : (memref<512xf32>) -> memref<512xf32>
    %105:3 = call @BatchNormGradOp75(%9, %arg80, %80#1, %104, %103) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %106 = call @ConvBackwardDataOp76(%105#0, %48) : (memref<32x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %107 = call @ConvBackwardFilterOp77(%79#0, %105#0) : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %108 = call @Unknown78(%81#2) : (memref<512xf32>) -> memref<512xf32>
    %109:3 = call @BatchNormGradOp79(%8, %arg89, %81#1, %108, %98) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %110 = call @ConvBackwardDataOp80(%109#0, %50) : (memref<32x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<32x256x14x14xf16>
    %111 = call @ConvBackwardFilterOp81(%79#0, %109#0) : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %112 = call @Unknown82(%110, %106, %79#1) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16>
    %113 = call @Unknown83(%78#2) : (memref<256xf32>) -> memref<256xf32>
    %114:3 = call @BatchNormGradOp84(%10, %arg75, %78#1, %113, %112) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %115 = call @ConvBackwardDataOp85(%114#0, %47) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %116 = call @ConvBackwardFilterOp86(%77#0, %114#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %117 = call @Unknown87(%77#1, %115) : (memref<32x256x14x14xi1>, memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16>
    %118 = call @Unknown88(%76#2) : (memref<256xf32>) -> memref<256xf32>
    %119:3 = call @BatchNormGradOp89(%11, %arg70, %76#1, %118, %117) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %120 = call @ConvBackwardDataOp90(%119#0, %46) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %121 = call @ConvBackwardFilterOp91(%75#0, %119#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %122 = call @Unknown92(%112, %120, %75#1) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16>
    %123 = call @Unknown93(%74#2) : (memref<256xf32>) -> memref<256xf32>
    %124:3 = call @BatchNormGradOp94(%12, %arg60, %74#1, %123, %122) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %125 = call @ConvBackwardDataOp95(%124#0, %44) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %126 = call @ConvBackwardFilterOp96(%73#0, %124#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %127 = call @Unknown97(%73#1, %125) : (memref<32x256x14x14xi1>, memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16>
    %128 = call @Unknown98(%71#2) : (memref<256xf32>) -> memref<256xf32>
    %129:3 = call @BatchNormGradOp99(%14, %arg55, %71#1, %128, %127) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %130 = call @ConvBackwardDataOp100(%129#0, %43) : (memref<32x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %131 = call @ConvBackwardFilterOp101(%70#0, %129#0) : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %132 = call @Unknown102(%72#2) : (memref<256xf32>) -> memref<256xf32>
    %133:3 = call @BatchNormGradOp103(%13, %arg64, %72#1, %132, %122) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %134 = call @ConvBackwardDataOp104(%133#0, %45) : (memref<32x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<32x128x28x28xf16>
    %135 = call @ConvBackwardFilterOp105(%70#0, %133#0) : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %136 = call @Unknown106(%134, %130, %70#1) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16>
    %137 = call @Unknown107(%69#2) : (memref<128xf32>) -> memref<128xf32>
    %138:3 = call @BatchNormGradOp108(%15, %arg50, %69#1, %137, %136) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %139 = call @ConvBackwardDataOp109(%138#0, %42) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %140 = call @ConvBackwardFilterOp110(%68#0, %138#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %141 = call @Unknown111(%68#1, %139) : (memref<32x128x28x28xi1>, memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16>
    %142 = call @Unknown112(%67#2) : (memref<128xf32>) -> memref<128xf32>
    %143:3 = call @BatchNormGradOp113(%16, %arg45, %67#1, %142, %141) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %144 = call @ConvBackwardDataOp114(%143#0, %41) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %145 = call @ConvBackwardFilterOp115(%66#0, %143#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %146 = call @Unknown116(%136, %144, %66#1) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16>
    %147 = call @Unknown117(%65#2) : (memref<128xf32>) -> memref<128xf32>
    %148:3 = call @BatchNormGradOp118(%17, %arg35, %65#1, %147, %146) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %149 = call @ConvBackwardDataOp119(%148#0, %39) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %150 = call @ConvBackwardFilterOp120(%64#0, %148#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %151 = call @Unknown121(%64#1, %149) : (memref<32x128x28x28xi1>, memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16>
    %152 = call @Unknown122(%62#2) : (memref<128xf32>) -> memref<128xf32>
    %153:3 = call @BatchNormGradOp123(%19, %arg30, %62#1, %152, %151) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %154 = call @ConvBackwardDataOp124(%153#0, %38) : (memref<32x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %155 = call @ConvBackwardFilterOp125(%61#0, %153#0) : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %156 = call @Unknown126(%63#2) : (memref<128xf32>) -> memref<128xf32>
    %157:3 = call @BatchNormGradOp127(%18, %arg39, %63#1, %156, %146) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %158 = call @ConvBackwardDataOp128(%157#0, %40) : (memref<32x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<32x64x56x56xf16>
    %159 = call @ConvBackwardFilterOp129(%61#0, %157#0) : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %160 = call @Unknown130(%158, %154, %61#1) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16>
    %161 = call @Unknown131(%60#2) : (memref<64xf32>) -> memref<64xf32>
    %162:3 = call @BatchNormGradOp132(%20, %arg25, %60#1, %161, %160) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %163 = call @ConvBackwardDataOp133(%162#0, %37) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %164 = call @ConvBackwardFilterOp134(%59#0, %162#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %165 = call @Unknown135(%59#1, %163) : (memref<32x64x56x56xi1>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    %166 = call @Unknown136(%58#2) : (memref<64xf32>) -> memref<64xf32>
    %167:3 = call @BatchNormGradOp137(%21, %arg20, %58#1, %166, %165) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %168 = call @ConvBackwardDataOp138(%167#0, %36) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %169 = call @ConvBackwardFilterOp139(%57#0, %167#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %170 = call @Unknown140(%160, %168, %57#1) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16>
    %171 = call @Unknown141(%56#2) : (memref<64xf32>) -> memref<64xf32>
    %172:3 = call @BatchNormGradOp142(%22, %arg15, %56#1, %171, %170) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %173 = call @ConvBackwardDataOp143(%172#0, %35) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %174 = call @ConvBackwardFilterOp144(%55#0, %172#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %175 = call @Unknown145(%55#1, %173) : (memref<32x64x56x56xi1>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    %176 = call @Unknown146(%54#2) : (memref<64xf32>) -> memref<64xf32>
    %177:3 = call @BatchNormGradOp147(%23, %arg10, %54#1, %176, %175) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %178 = call @ConvBackwardDataOp148(%177#0, %34) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %179 = call @ConvBackwardFilterOp149(%24, %177#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %180 = call @Unknown150(%170, %178) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    "lmhlo.select_and_scatter"(%53#0, %180, %28, %4) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %252 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%252) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %252 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%252) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<32x64x112x112xf16>, memref<32x64x56x56xf16>, memref<f16>, memref<32x64x112x112xf16>) -> ()
    %181 = call @Unknown151(%53#1, %4) : (memref<32x64x112x112xi1>, memref<32x64x112x112xf16>) -> memref<32x64x112x112xf16>
    %182 = call @Unknown152(%32#2) : (memref<64xf32>) -> memref<64xf32>
    %183:3 = call @BatchNormGradOp153(%26, %arg5, %32#1, %182, %181) : (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %184 = call @ConvBackwardFilterOp154(%30, %183#0) : (memref<32x3x224x224xf16>, memref<32x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %185 = call @Unknown155(%184) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %186 = call @Unknown156(%179) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %187 = call @Unknown157(%174) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %188 = call @Unknown158(%169) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %189 = call @Unknown159(%164) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %190 = call @Unknown160(%155) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %191 = call @Unknown161(%150) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %192 = call @Unknown162(%159) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %193 = call @Unknown163(%145) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %194 = call @Unknown164(%140) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %195 = call @Unknown165(%131) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %196 = call @Unknown166(%126) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %197 = call @Unknown167(%135) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %198 = call @Unknown168(%121) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %199 = call @Unknown169(%116) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %200 = call @Unknown170(%107) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %201 = call @Unknown171(%102) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %202 = call @Unknown172(%111) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %203 = call @Unknown173(%97) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %204 = call @Unknown174(%92) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    "lmhlo.reduce"(%88#0, %28, %3) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<32x512x7x7xf16>, memref<f16>, memref<32x512xf16>) -> ()
    %205 = call @Unknown175(%3) : (memref<32x512xf16>) -> memref<32x512xf16>
    %206 = call @MatmulOp176(%205, %arg102) : (memref<32x512xf16>, memref<32x1000xf16>) -> memref<1000x512xf16>
    %207 = call @Unknown177(%206) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %208 = call @Unknown178(%arg102) : (memref<32x1000xf16>) -> memref<32x1000xf32>
    "lmhlo.reduce"(%208, %29, %2) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<32x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %209 = call @Unknown179(%2) : (memref<1000xf32>) -> memref<1000xf32>
    %210 = call @Unknown180(%32#1, %arg3) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %211 = call @Unknown181(%32#2, %arg2) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %212 = call @Unknown182(%54#1, %arg8) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %213 = call @Unknown183(%54#2, %arg7) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %214 = call @Unknown184(%56#1, %arg13) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %215 = call @Unknown185(%56#2, %arg12) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %216 = call @Unknown186(%58#1, %arg18) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %217 = call @Unknown187(%58#2, %arg17) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %218 = call @Unknown188(%60#1, %arg23) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %219 = call @Unknown189(%60#2, %arg22) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %220 = call @Unknown190(%62#1, %arg28) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %221 = call @Unknown191(%62#2, %arg27) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %222 = call @Unknown192(%65#1, %arg33) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %223 = call @Unknown193(%65#2, %arg32) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %224 = call @Unknown194(%63#1, %arg37) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %225 = call @Unknown195(%63#2, %arg36) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %226 = call @Unknown196(%67#1, %arg43) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %227 = call @Unknown197(%67#2, %arg42) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %228 = call @Unknown198(%69#1, %arg48) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %229 = call @Unknown199(%69#2, %arg47) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %230 = call @Unknown200(%71#1, %arg53) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %231 = call @Unknown201(%71#2, %arg52) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %232 = call @Unknown202(%74#1, %arg58) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %233 = call @Unknown203(%74#2, %arg57) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %234 = call @Unknown204(%72#1, %arg62) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %235 = call @Unknown205(%72#2, %arg61) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %236 = call @Unknown206(%76#1, %arg68) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %237 = call @Unknown207(%76#2, %arg67) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %238 = call @Unknown208(%78#1, %arg73) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %239 = call @Unknown209(%78#2, %arg72) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %240 = call @Unknown210(%80#1, %arg78) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %241 = call @Unknown211(%80#2, %arg77) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %242 = call @Unknown212(%83#1, %arg83) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %243 = call @Unknown213(%83#2, %arg82) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %244 = call @Unknown214(%81#1, %arg87) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %245 = call @Unknown215(%81#2, %arg86) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %246 = call @Unknown216(%85#1, %arg93) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %247 = call @Unknown217(%85#2, %arg92) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %248 = call @Unknown218(%87#1, %arg98) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %249 = call @Unknown219(%87#2, %arg97) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %250 = call @Unknown220(%arg103) : (memref<1000xf32>) -> memref<1000xf16>
    "lmhlo.dot"(%205, %33, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512xf16>, memref<1000x512xf16>, memref<32x1000xf16>) -> ()
    %251 = call @Unknown221(%250, %1) : (memref<1000xf16>, memref<32x1000xf16>) -> memref<32x1000xf16>
    return %185, %183#1, %183#2, %186, %177#1, %177#2, %187, %172#1, %172#2, %188, %167#1, %167#2, %189, %162#1, %162#2, %190, %153#1, %153#2, %191, %148#1, %148#2, %192, %157#1, %157#2, %193, %143#1, %143#2, %194, %138#1, %138#2, %195, %129#1, %129#2, %196, %124#1, %124#2, %197, %133#1, %133#2, %198, %119#1, %119#2, %199, %114#1, %114#2, %200, %105#1, %105#2, %201, %100#1, %100#2, %202, %109#1, %109#2, %203, %95#1, %95#2, %204, %90#1, %90#2, %207, %209, %210, %211, %0, %212, %213, %0, %214, %215, %0, %216, %217, %0, %218, %219, %0, %220, %221, %0, %222, %223, %0, %224, %225, %0, %226, %227, %0, %228, %229, %0, %230, %231, %0, %232, %233, %0, %234, %235, %0, %236, %237, %0, %238, %239, %0, %240, %241, %0, %242, %243, %0, %244, %245, %0, %246, %247, %0, %248, %249, %0, %251 : memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<32x1000xf16>
  }
}

