// RUN: byteir-opt %s -byre-host="device-file-name=your_file" | FileCheck %s

// CHECK-LABEL: func @main
module attributes {byre.container_module, gpu.container_module} {
  gpu.module @Unknown0_kernel {
    gpu.func @Unknown0_kernel(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x3x224x224xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x3x224x224xf32>
        %37 = arith.truncf %36 : f32 to f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x3x224x224xf16>
      }
      gpu.return
    }
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
  gpu.module @Unknown3_kernel {
    gpu.func @Unknown3_kernel(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) kernel {
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x112x112xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
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
  gpu.module @Unknown6_kernel {
    gpu.func @Unknown6_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
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
  gpu.module @Unknown9_kernel {
    gpu.func @Unknown9_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown10_kernel {
    gpu.func @Unknown10_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
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
  gpu.module @Unknown12_kernel {
    gpu.func @Unknown12_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown13_kernel {
    gpu.func @Unknown13_kernel(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
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
  gpu.module @Unknown15_kernel {
    gpu.func @Unknown15_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown16_kernel {
    gpu.func @Unknown16_kernel(%arg0: memref<128x64x1x1xf32>, %arg1: memref<128x64x1x1xf16>) kernel {
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
  gpu.module @Unknown18_kernel {
    gpu.func @Unknown18_kernel(%arg0: memref<128x64x3x3xf32>, %arg1: memref<128x64x3x3xf16>) kernel {
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
  gpu.module @Unknown20_kernel {
    gpu.func @Unknown20_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown21_kernel {
    gpu.func @Unknown21_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
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
  gpu.module @Unknown23_kernel {
    gpu.func @Unknown23_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown24_kernel {
    gpu.func @Unknown24_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
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
  gpu.module @Unknown26_kernel {
    gpu.func @Unknown26_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown27_kernel {
    gpu.func @Unknown27_kernel(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
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
  gpu.module @Unknown29_kernel {
    gpu.func @Unknown29_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown30_kernel {
    gpu.func @Unknown30_kernel(%arg0: memref<256x128x1x1xf32>, %arg1: memref<256x128x1x1xf16>) kernel {
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
  gpu.module @Unknown32_kernel {
    gpu.func @Unknown32_kernel(%arg0: memref<256x128x3x3xf32>, %arg1: memref<256x128x3x3xf16>) kernel {
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
  gpu.module @Unknown34_kernel {
    gpu.func @Unknown34_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown35_kernel {
    gpu.func @Unknown35_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
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
  gpu.module @Unknown37_kernel {
    gpu.func @Unknown37_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown38_kernel {
    gpu.func @Unknown38_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
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
  gpu.module @Unknown40_kernel {
    gpu.func @Unknown40_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown41_kernel {
    gpu.func @Unknown41_kernel(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
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
  gpu.module @Unknown43_kernel {
    gpu.func @Unknown43_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown44_kernel {
    gpu.func @Unknown44_kernel(%arg0: memref<512x256x1x1xf32>, %arg1: memref<512x256x1x1xf16>) kernel {
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
  gpu.module @Unknown46_kernel {
    gpu.func @Unknown46_kernel(%arg0: memref<512x256x3x3xf32>, %arg1: memref<512x256x3x3xf16>) kernel {
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
  gpu.module @Unknown48_kernel {
    gpu.func @Unknown48_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown49_kernel {
    gpu.func @Unknown49_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
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
  gpu.module @Unknown51_kernel {
    gpu.func @Unknown51_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown52_kernel {
    gpu.func @Unknown52_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
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
  gpu.module @Unknown54_kernel {
    gpu.func @Unknown54_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = arith.maxf %36, %cst : f16
        memref.store %37, %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown55_kernel {
    gpu.func @Unknown55_kernel(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
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
  gpu.module @Unknown57_kernel {
    gpu.func @Unknown57_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %36 = memref.load %arg0[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %38 = arith.addf %36, %37 : f16
        %39 = arith.maxf %38, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown58_kernel {
    gpu.func @Unknown58_kernel(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
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
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c512 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<1x512xf16>
        %17 = arith.mulf %16, %cst : f16
        memref.store %17, %arg1[%15, %9] : memref<1x512xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown59_kernel {
    gpu.func @Unknown59_kernel(%arg0: memref<1000x512xf32>, %arg1: memref<1000x512xf16>) kernel {
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
  gpu.module @Unknown60_kernel {
    gpu.func @Unknown60_kernel(%arg0: memref<1000xf32>, %arg1: memref<1000xf16>, %arg2: memref<1x1000xf16>, %arg3: memref<1x1000xf16>) kernel {
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
        %8 = arith.remsi %4, %c1000 : index
        %c0 = arith.constant 0 : index
        %9 = arith.cmpi slt, %8, %c0 : index
        %10 = arith.addi %8, %c1000 : index
        %11 = select %9, %10, %8 : index
        %c-1 = arith.constant -1 : index
        %12 = arith.cmpi slt, %4, %c0 : index
        %13 = arith.subi %c-1, %4 : index
        %14 = select %12, %13, %4 : index
        %15 = arith.divsi %14, %c1000 : index
        %16 = arith.subi %c-1, %15 : index
        %17 = select %12, %16, %15 : index
        %18 = memref.load %arg2[%17, %11] : memref<1x1000xf16>
        %19 = memref.load %arg1[%11] : memref<1000xf16>
        %20 = arith.addf %18, %19 : f16
        memref.store %20, %arg3[%17, %11] : memref<1x1000xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown61_kernel {
    gpu.func @Unknown61_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown62_kernel {
    gpu.func @Unknown62_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown63_kernel {
    gpu.func @Unknown63_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown64_kernel {
    gpu.func @Unknown64_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown65_kernel {
    gpu.func @Unknown65_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown66_kernel {
    gpu.func @Unknown66_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown67_kernel {
    gpu.func @Unknown67_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown68_kernel {
    gpu.func @Unknown68_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown69_kernel {
    gpu.func @Unknown69_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown70_kernel {
    gpu.func @Unknown70_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
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
  gpu.module @Unknown71_kernel {
    gpu.func @Unknown71_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown72_kernel {
    gpu.func @Unknown72_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown73_kernel {
    gpu.func @Unknown73_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown74_kernel {
    gpu.func @Unknown74_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown75_kernel {
    gpu.func @Unknown75_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown76_kernel {
    gpu.func @Unknown76_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown77_kernel {
    gpu.func @Unknown77_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown78_kernel {
    gpu.func @Unknown78_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown79_kernel {
    gpu.func @Unknown79_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown80_kernel {
    gpu.func @Unknown80_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
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
  gpu.module @Unknown81_kernel {
    gpu.func @Unknown81_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown82_kernel {
    gpu.func @Unknown82_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown83_kernel {
    gpu.func @Unknown83_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown84_kernel {
    gpu.func @Unknown84_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown85_kernel {
    gpu.func @Unknown85_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown86_kernel {
    gpu.func @Unknown86_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown87_kernel {
    gpu.func @Unknown87_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown88_kernel {
    gpu.func @Unknown88_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown89_kernel {
    gpu.func @Unknown89_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown90_kernel {
    gpu.func @Unknown90_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
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
  gpu.module @Unknown91_kernel {
    gpu.func @Unknown91_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown92_kernel {
    gpu.func @Unknown92_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown93_kernel {
    gpu.func @Unknown93_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown94_kernel {
    gpu.func @Unknown94_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown95_kernel {
    gpu.func @Unknown95_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown96_kernel {
    gpu.func @Unknown96_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown97_kernel {
    gpu.func @Unknown97_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown98_kernel {
    gpu.func @Unknown98_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown99_kernel {
    gpu.func @Unknown99_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  gpu.module @Unknown100_kernel {
    gpu.func @Unknown100_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
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
  func @main(%arg0: memref<64xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<64xf32> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<64x3x7x7xf32> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1000xf32> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<1000x512xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<64xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<64xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<64xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<64xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<64x64x3x3xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<64x64x3x3xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<64xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<64xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<64xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<64xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<64x64x3x3xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<64x64x3x3xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<128xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x64x3x3xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128x128x3x3xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128x64x1x1xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128x128x3x3xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128x3x3xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<256xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<256xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<256xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<256xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<256x128x3x3xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<256x256x3x3xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<256x128x1x1xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<256xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<256xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<256xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<256xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<256xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<256xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<256x256x3x3xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<256x256x3x3xf32> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<512xf32> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<512xf32> {byre.argname = "Input48", byre.argtype = 1 : i32}, %arg49: memref<512xf32> {byre.argname = "Input49", byre.argtype = 1 : i32}, %arg50: memref<512xf32> {byre.argname = "Input50", byre.argtype = 1 : i32}, %arg51: memref<512x256x3x3xf32> {byre.argname = "Input51", byre.argtype = 1 : i32}, %arg52: memref<512x512x3x3xf32> {byre.argname = "Input52", byre.argtype = 1 : i32}, %arg53: memref<512x256x1x1xf32> {byre.argname = "Input53", byre.argtype = 1 : i32}, %arg54: memref<512xf32> {byre.argname = "Input54", byre.argtype = 1 : i32}, %arg55: memref<512xf32> {byre.argname = "Input55", byre.argtype = 1 : i32}, %arg56: memref<512xf32> {byre.argname = "Input56", byre.argtype = 1 : i32}, %arg57: memref<512xf32> {byre.argname = "Input57", byre.argtype = 1 : i32}, %arg58: memref<512xf32> {byre.argname = "Input58", byre.argtype = 1 : i32}, %arg59: memref<512xf32> {byre.argname = "Input59", byre.argtype = 1 : i32}, %arg60: memref<512x512x3x3xf32> {byre.argname = "Input60", byre.argtype = 1 : i32}, %arg61: memref<512x512x3x3xf32> {byre.argname = "Input61", byre.argtype = 1 : i32}, %arg62: memref<i64> {byre.argname = "Input62", byre.argtype = 1 : i32}, %arg63: memref<64xf32> {byre.argname = "Input63", byre.argtype = 1 : i32}, %arg64: memref<64xf32> {byre.argname = "Input64", byre.argtype = 1 : i32}, %arg65: memref<i64> {byre.argname = "Input65", byre.argtype = 1 : i32}, %arg66: memref<64xf32> {byre.argname = "Input66", byre.argtype = 1 : i32}, %arg67: memref<64xf32> {byre.argname = "Input67", byre.argtype = 1 : i32}, %arg68: memref<i64> {byre.argname = "Input68", byre.argtype = 1 : i32}, %arg69: memref<64xf32> {byre.argname = "Input69", byre.argtype = 1 : i32}, %arg70: memref<64xf32> {byre.argname = "Input70", byre.argtype = 1 : i32}, %arg71: memref<i64> {byre.argname = "Input71", byre.argtype = 1 : i32}, %arg72: memref<64xf32> {byre.argname = "Input72", byre.argtype = 1 : i32}, %arg73: memref<64xf32> {byre.argname = "Input73", byre.argtype = 1 : i32}, %arg74: memref<i64> {byre.argname = "Input74", byre.argtype = 1 : i32}, %arg75: memref<64xf32> {byre.argname = "Input75", byre.argtype = 1 : i32}, %arg76: memref<64xf32> {byre.argname = "Input76", byre.argtype = 1 : i32}, %arg77: memref<i64> {byre.argname = "Input77", byre.argtype = 1 : i32}, %arg78: memref<128xf32> {byre.argname = "Input78", byre.argtype = 1 : i32}, %arg79: memref<128xf32> {byre.argname = "Input79", byre.argtype = 1 : i32}, %arg80: memref<i64> {byre.argname = "Input80", byre.argtype = 1 : i32}, %arg81: memref<128xf32> {byre.argname = "Input81", byre.argtype = 1 : i32}, %arg82: memref<128xf32> {byre.argname = "Input82", byre.argtype = 1 : i32}, %arg83: memref<i64> {byre.argname = "Input83", byre.argtype = 1 : i32}, %arg84: memref<128xf32> {byre.argname = "Input84", byre.argtype = 1 : i32}, %arg85: memref<128xf32> {byre.argname = "Input85", byre.argtype = 1 : i32}, %arg86: memref<i64> {byre.argname = "Input86", byre.argtype = 1 : i32}, %arg87: memref<128xf32> {byre.argname = "Input87", byre.argtype = 1 : i32}, %arg88: memref<128xf32> {byre.argname = "Input88", byre.argtype = 1 : i32}, %arg89: memref<i64> {byre.argname = "Input89", byre.argtype = 1 : i32}, %arg90: memref<128xf32> {byre.argname = "Input90", byre.argtype = 1 : i32}, %arg91: memref<128xf32> {byre.argname = "Input91", byre.argtype = 1 : i32}, %arg92: memref<i64> {byre.argname = "Input92", byre.argtype = 1 : i32}, %arg93: memref<256xf32> {byre.argname = "Input93", byre.argtype = 1 : i32}, %arg94: memref<256xf32> {byre.argname = "Input94", byre.argtype = 1 : i32}, %arg95: memref<i64> {byre.argname = "Input95", byre.argtype = 1 : i32}, %arg96: memref<256xf32> {byre.argname = "Input96", byre.argtype = 1 : i32}, %arg97: memref<256xf32> {byre.argname = "Input97", byre.argtype = 1 : i32}, %arg98: memref<i64> {byre.argname = "Input98", byre.argtype = 1 : i32}, %arg99: memref<256xf32> {byre.argname = "Input99", byre.argtype = 1 : i32}, %arg100: memref<256xf32> {byre.argname = "Input100", byre.argtype = 1 : i32}, %arg101: memref<i64> {byre.argname = "Input101", byre.argtype = 1 : i32}, %arg102: memref<256xf32> {byre.argname = "Input102", byre.argtype = 1 : i32}, %arg103: memref<256xf32> {byre.argname = "Input103", byre.argtype = 1 : i32}, %arg104: memref<i64> {byre.argname = "Input104", byre.argtype = 1 : i32}, %arg105: memref<256xf32> {byre.argname = "Input105", byre.argtype = 1 : i32}, %arg106: memref<256xf32> {byre.argname = "Input106", byre.argtype = 1 : i32}, %arg107: memref<i64> {byre.argname = "Input107", byre.argtype = 1 : i32}, %arg108: memref<512xf32> {byre.argname = "Input108", byre.argtype = 1 : i32}, %arg109: memref<512xf32> {byre.argname = "Input109", byre.argtype = 1 : i32}, %arg110: memref<i64> {byre.argname = "Input110", byre.argtype = 1 : i32}, %arg111: memref<512xf32> {byre.argname = "Input111", byre.argtype = 1 : i32}, %arg112: memref<512xf32> {byre.argname = "Input112", byre.argtype = 1 : i32}, %arg113: memref<i64> {byre.argname = "Input113", byre.argtype = 1 : i32}, %arg114: memref<512xf32> {byre.argname = "Input114", byre.argtype = 1 : i32}, %arg115: memref<512xf32> {byre.argname = "Input115", byre.argtype = 1 : i32}, %arg116: memref<i64> {byre.argname = "Input116", byre.argtype = 1 : i32}, %arg117: memref<512xf32> {byre.argname = "Input117", byre.argtype = 1 : i32}, %arg118: memref<512xf32> {byre.argname = "Input118", byre.argtype = 1 : i32}, %arg119: memref<i64> {byre.argname = "Input119", byre.argtype = 1 : i32}, %arg120: memref<512xf32> {byre.argname = "Input120", byre.argtype = 1 : i32}, %arg121: memref<512xf32> {byre.argname = "Input121", byre.argtype = 1 : i32}, %arg122: memref<1x3x224x224xf32> {byre.argname = "Input122", byre.argtype = 1 : i32}, %arg123: memref<1x1000xf16> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg124: memref<64xf32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg125: memref<64xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg126: memref<64xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg127: memref<64xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg128: memref<64xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg129: memref<64xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg130: memref<64xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg131: memref<64xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg132: memref<64xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg133: memref<64xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg134: memref<128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg135: memref<128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg136: memref<128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg137: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg138: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg139: memref<128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg140: memref<128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg141: memref<128xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg142: memref<128xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg143: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg144: memref<256xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg145: memref<256xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg146: memref<256xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg147: memref<256xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg148: memref<256xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg149: memref<256xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg150: memref<256xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg151: memref<256xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg152: memref<256xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg153: memref<256xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg154: memref<512xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg155: memref<512xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg156: memref<512xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg157: memref<512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg158: memref<512xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg159: memref<512xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg160: memref<512xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg161: memref<512xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg162: memref<512xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg163: memref<512xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg164: memref<64xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg165: memref<64xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg166: memref<64xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}, %arg167: memref<64xf32> {byre.argname = "Output44", byre.argtype = 2 : i32}, %arg168: memref<64xf32> {byre.argname = "Output45", byre.argtype = 2 : i32}, %arg169: memref<64xf32> {byre.argname = "Output46", byre.argtype = 2 : i32}, %arg170: memref<64xf32> {byre.argname = "Output47", byre.argtype = 2 : i32}, %arg171: memref<64xf32> {byre.argname = "Output48", byre.argtype = 2 : i32}, %arg172: memref<64xf32> {byre.argname = "Output49", byre.argtype = 2 : i32}, %arg173: memref<64xf32> {byre.argname = "Output50", byre.argtype = 2 : i32}, %arg174: memref<128xf32> {byre.argname = "Output51", byre.argtype = 2 : i32}, %arg175: memref<128xf32> {byre.argname = "Output52", byre.argtype = 2 : i32}, %arg176: memref<128xf32> {byre.argname = "Output53", byre.argtype = 2 : i32}, %arg177: memref<128xf32> {byre.argname = "Output54", byre.argtype = 2 : i32}, %arg178: memref<128xf32> {byre.argname = "Output55", byre.argtype = 2 : i32}, %arg179: memref<128xf32> {byre.argname = "Output56", byre.argtype = 2 : i32}, %arg180: memref<128xf32> {byre.argname = "Output57", byre.argtype = 2 : i32}, %arg181: memref<128xf32> {byre.argname = "Output58", byre.argtype = 2 : i32}, %arg182: memref<128xf32> {byre.argname = "Output59", byre.argtype = 2 : i32}, %arg183: memref<128xf32> {byre.argname = "Output60", byre.argtype = 2 : i32}, %arg184: memref<256xf32> {byre.argname = "Output61", byre.argtype = 2 : i32}, %arg185: memref<256xf32> {byre.argname = "Output62", byre.argtype = 2 : i32}, %arg186: memref<256xf32> {byre.argname = "Output63", byre.argtype = 2 : i32}, %arg187: memref<256xf32> {byre.argname = "Output64", byre.argtype = 2 : i32}, %arg188: memref<256xf32> {byre.argname = "Output65", byre.argtype = 2 : i32}, %arg189: memref<256xf32> {byre.argname = "Output66", byre.argtype = 2 : i32}, %arg190: memref<256xf32> {byre.argname = "Output67", byre.argtype = 2 : i32}, %arg191: memref<256xf32> {byre.argname = "Output68", byre.argtype = 2 : i32}, %arg192: memref<256xf32> {byre.argname = "Output69", byre.argtype = 2 : i32}, %arg193: memref<256xf32> {byre.argname = "Output70", byre.argtype = 2 : i32}, %arg194: memref<512xf32> {byre.argname = "Output71", byre.argtype = 2 : i32}, %arg195: memref<512xf32> {byre.argname = "Output72", byre.argtype = 2 : i32}, %arg196: memref<512xf32> {byre.argname = "Output73", byre.argtype = 2 : i32}, %arg197: memref<512xf32> {byre.argname = "Output74", byre.argtype = 2 : i32}, %arg198: memref<512xf32> {byre.argname = "Output75", byre.argtype = 2 : i32}, %arg199: memref<512xf32> {byre.argname = "Output76", byre.argtype = 2 : i32}, %arg200: memref<512xf32> {byre.argname = "Output77", byre.argtype = 2 : i32}, %arg201: memref<512xf32> {byre.argname = "Output78", byre.argtype = 2 : i32}, %arg202: memref<512xf32> {byre.argname = "Output79", byre.argtype = 2 : i32}, %arg203: memref<512xf32> {byre.argname = "Output80", byre.argtype = 2 : i32}, %arg204: memref<64x3x7x7xf16> {byre.argname = "Output81", byre.argtype = 2 : i32}, %arg205: memref<1x3x224x224xf16> {byre.argname = "Output82", byre.argtype = 2 : i32}, %arg206: memref<1x64x112x112xf16> {byre.argname = "Output83", byre.argtype = 2 : i32}, %arg207: memref<1x64x112x112xf16> {byre.argname = "Output84", byre.argtype = 2 : i32}, %arg208: memref<1x64x56x56xf16> {byre.argname = "Output85", byre.argtype = 2 : i32}, %arg209: memref<64x64x3x3xf16> {byre.argname = "Output86", byre.argtype = 2 : i32}, %arg210: memref<1x64x56x56xf16> {byre.argname = "Output87", byre.argtype = 2 : i32}, %arg211: memref<1x64x56x56xf16> {byre.argname = "Output88", byre.argtype = 2 : i32}, %arg212: memref<64x64x3x3xf16> {byre.argname = "Output89", byre.argtype = 2 : i32}, %arg213: memref<1x64x56x56xf16> {byre.argname = "Output90", byre.argtype = 2 : i32}, %arg214: memref<1x64x56x56xf16> {byre.argname = "Output91", byre.argtype = 2 : i32}, %arg215: memref<64x64x3x3xf16> {byre.argname = "Output92", byre.argtype = 2 : i32}, %arg216: memref<1x64x56x56xf16> {byre.argname = "Output93", byre.argtype = 2 : i32}, %arg217: memref<1x64x56x56xf16> {byre.argname = "Output94", byre.argtype = 2 : i32}, %arg218: memref<64x64x3x3xf16> {byre.argname = "Output95", byre.argtype = 2 : i32}, %arg219: memref<1x64x56x56xf16> {byre.argname = "Output96", byre.argtype = 2 : i32}, %arg220: memref<1x64x56x56xf16> {byre.argname = "Output97", byre.argtype = 2 : i32}, %arg221: memref<128x64x3x3xf16> {byre.argname = "Output98", byre.argtype = 2 : i32}, %arg222: memref<1x128x28x28xf16> {byre.argname = "Output99", byre.argtype = 2 : i32}, %arg223: memref<1x128x28x28xf16> {byre.argname = "Output100", byre.argtype = 2 : i32}, %arg224: memref<128x128x3x3xf16> {byre.argname = "Output101", byre.argtype = 2 : i32}, %arg225: memref<1x128x28x28xf16> {byre.argname = "Output102", byre.argtype = 2 : i32}, %arg226: memref<128x64x1x1xf16> {byre.argname = "Output103", byre.argtype = 2 : i32}, %arg227: memref<1x128x28x28xf16> {byre.argname = "Output104", byre.argtype = 2 : i32}, %arg228: memref<1x128x28x28xf16> {byre.argname = "Output105", byre.argtype = 2 : i32}, %arg229: memref<128x128x3x3xf16> {byre.argname = "Output106", byre.argtype = 2 : i32}, %arg230: memref<1x128x28x28xf16> {byre.argname = "Output107", byre.argtype = 2 : i32}, %arg231: memref<1x128x28x28xf16> {byre.argname = "Output108", byre.argtype = 2 : i32}, %arg232: memref<128x128x3x3xf16> {byre.argname = "Output109", byre.argtype = 2 : i32}, %arg233: memref<1x128x28x28xf16> {byre.argname = "Output110", byre.argtype = 2 : i32}, %arg234: memref<1x128x28x28xf16> {byre.argname = "Output111", byre.argtype = 2 : i32}, %arg235: memref<256x128x3x3xf16> {byre.argname = "Output112", byre.argtype = 2 : i32}, %arg236: memref<1x256x14x14xf16> {byre.argname = "Output113", byre.argtype = 2 : i32}, %arg237: memref<1x256x14x14xf16> {byre.argname = "Output114", byre.argtype = 2 : i32}, %arg238: memref<256x256x3x3xf16> {byre.argname = "Output115", byre.argtype = 2 : i32}, %arg239: memref<1x256x14x14xf16> {byre.argname = "Output116", byre.argtype = 2 : i32}, %arg240: memref<256x128x1x1xf16> {byre.argname = "Output117", byre.argtype = 2 : i32}, %arg241: memref<1x256x14x14xf16> {byre.argname = "Output118", byre.argtype = 2 : i32}, %arg242: memref<1x256x14x14xf16> {byre.argname = "Output119", byre.argtype = 2 : i32}, %arg243: memref<256x256x3x3xf16> {byre.argname = "Output120", byre.argtype = 2 : i32}, %arg244: memref<1x256x14x14xf16> {byre.argname = "Output121", byre.argtype = 2 : i32}, %arg245: memref<1x256x14x14xf16> {byre.argname = "Output122", byre.argtype = 2 : i32}, %arg246: memref<256x256x3x3xf16> {byre.argname = "Output123", byre.argtype = 2 : i32}, %arg247: memref<1x256x14x14xf16> {byre.argname = "Output124", byre.argtype = 2 : i32}, %arg248: memref<1x256x14x14xf16> {byre.argname = "Output125", byre.argtype = 2 : i32}, %arg249: memref<512x256x3x3xf16> {byre.argname = "Output126", byre.argtype = 2 : i32}, %arg250: memref<1x512x7x7xf16> {byre.argname = "Output127", byre.argtype = 2 : i32}, %arg251: memref<1x512x7x7xf16> {byre.argname = "Output128", byre.argtype = 2 : i32}, %arg252: memref<512x512x3x3xf16> {byre.argname = "Output129", byre.argtype = 2 : i32}, %arg253: memref<1x512x7x7xf16> {byre.argname = "Output130", byre.argtype = 2 : i32}, %arg254: memref<512x256x1x1xf16> {byre.argname = "Output131", byre.argtype = 2 : i32}, %arg255: memref<1x512x7x7xf16> {byre.argname = "Output132", byre.argtype = 2 : i32}, %arg256: memref<1x512x7x7xf16> {byre.argname = "Output133", byre.argtype = 2 : i32}, %arg257: memref<512x512x3x3xf16> {byre.argname = "Output134", byre.argtype = 2 : i32}, %arg258: memref<1x512x7x7xf16> {byre.argname = "Output135", byre.argtype = 2 : i32}, %arg259: memref<1x512x7x7xf16> {byre.argname = "Output136", byre.argtype = 2 : i32}, %arg260: memref<512x512x3x3xf16> {byre.argname = "Output137", byre.argtype = 2 : i32}, %arg261: memref<1x512x7x7xf16> {byre.argname = "Output138", byre.argtype = 2 : i32}, %arg262: memref<1x512x7x7xf16> {byre.argname = "Output139", byre.argtype = 2 : i32}, %arg263: memref<1x512xf16> {byre.argname = "Output140", byre.argtype = 2 : i32}, %arg264: memref<512x1000xf16> {byre.argname = "Output141", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<1x1000xf16>
    %1 = memref.alloc() : memref<1x512xf16>
    byre.compute @PTXOp(%arg122, %arg205) {BlockSize.x = 32 : i32, GridSize.x = 4704 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown0_kernel"} : memref<1x3x224x224xf32>, memref<1x3x224x224xf16>
    byre.compute @PTXOp(%arg2, %arg204) {BlockSize.x = 32 : i32, GridSize.x = 294 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown1_kernel"} : memref<64x3x7x7xf32>, memref<64x3x7x7xf16>
    byre.compute @ConvOpf16f16f16(%arg205, %arg204, %arg206) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16>, memref<64x3x7x7xf16>, memref<1x64x112x112xf16>
    %2 = memref.alloc() : memref<1x64x112x112xf16>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg206, %arg125, %arg124, %2, %3, %4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%2, %arg207) {BlockSize.x = 32 : i32, GridSize.x = 25088 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3_kernel"} : memref<1x64x112x112xf16>, memref<1x64x112x112xf16>
    byre.compute @PoolMaxOpf16f16(%arg207, %arg208) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16>, memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg9, %arg209) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown4_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg208, %arg209, %arg210) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %5 = memref.alloc() : memref<1x64x56x56xf16>
    %6 = memref.alloc() : memref<64xf32>
    %7 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg210, %arg127, %arg126, %5, %6, %7) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%5, %arg211) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown6_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg10, %arg212) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown7_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg211, %arg212, %arg213) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %8 = memref.alloc() : memref<1x64x56x56xf16>
    %9 = memref.alloc() : memref<64xf32>
    %10 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg213, %arg129, %arg128, %8, %9, %10) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%8, %arg208, %arg214) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown9_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg15, %arg215) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown10_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg214, %arg215, %arg216) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %11 = memref.alloc() : memref<1x64x56x56xf16>
    %12 = memref.alloc() : memref<64xf32>
    %13 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg216, %arg131, %arg130, %11, %12, %13) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%11, %arg217) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown12_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg16, %arg218) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown13_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg217, %arg218, %arg219) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %14 = memref.alloc() : memref<1x64x56x56xf16>
    %15 = memref.alloc() : memref<64xf32>
    %16 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg219, %arg133, %arg132, %14, %15, %16) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%14, %arg214, %arg220) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown15_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg23, %arg226) {BlockSize.x = 32 : i32, GridSize.x = 256 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown16_kernel"} : memref<128x64x1x1xf32>, memref<128x64x1x1xf16>
    byre.compute @ConvOpf16f16f16(%arg220, %arg226, %arg227) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>
    %17 = memref.alloc() : memref<1x128x28x28xf16>
    %18 = memref.alloc() : memref<128xf32>
    %19 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg227, %arg139, %arg138, %17, %18, %19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%arg21, %arg221) {BlockSize.x = 32 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown18_kernel"} : memref<128x64x3x3xf32>, memref<128x64x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg220, %arg221, %arg222) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>
    %20 = memref.alloc() : memref<1x128x28x28xf16>
    %21 = memref.alloc() : memref<128xf32>
    %22 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg222, %arg135, %arg134, %20, %21, %22) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%20, %arg223) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown20_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg22, %arg224) {BlockSize.x = 32 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown21_kernel"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg223, %arg224, %arg225) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %23 = memref.alloc() : memref<1x128x28x28xf16>
    %24 = memref.alloc() : memref<128xf32>
    %25 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg225, %arg137, %arg136, %23, %24, %25) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%23, %17, %arg228) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown23_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg30, %arg229) {BlockSize.x = 32 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown24_kernel"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg228, %arg229, %arg230) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %26 = memref.alloc() : memref<1x128x28x28xf16>
    %27 = memref.alloc() : memref<128xf32>
    %28 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg230, %arg141, %arg140, %26, %27, %28) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%26, %arg231) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown26_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg31, %arg232) {BlockSize.x = 32 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown27_kernel"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg231, %arg232, %arg233) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %29 = memref.alloc() : memref<1x128x28x28xf16>
    %30 = memref.alloc() : memref<128xf32>
    %31 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg233, %arg143, %arg142, %29, %30, %31) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%29, %arg228, %arg234) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown29_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg38, %arg240) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown30_kernel"} : memref<256x128x1x1xf32>, memref<256x128x1x1xf16>
    byre.compute @ConvOpf16f16f16(%arg234, %arg240, %arg241) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>
    %32 = memref.alloc() : memref<1x256x14x14xf16>
    %33 = memref.alloc() : memref<256xf32>
    %34 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg241, %arg149, %arg148, %32, %33, %34) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%arg36, %arg235) {BlockSize.x = 32 : i32, GridSize.x = 9216 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown32_kernel"} : memref<256x128x3x3xf32>, memref<256x128x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg234, %arg235, %arg236) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>
    %35 = memref.alloc() : memref<1x256x14x14xf16>
    %36 = memref.alloc() : memref<256xf32>
    %37 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg236, %arg145, %arg144, %35, %36, %37) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%35, %arg237) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown34_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg37, %arg238) {BlockSize.x = 32 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown35_kernel"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg237, %arg238, %arg239) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %38 = memref.alloc() : memref<1x256x14x14xf16>
    %39 = memref.alloc() : memref<256xf32>
    %40 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg239, %arg147, %arg146, %38, %39, %40) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%38, %32, %arg242) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown37_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg45, %arg243) {BlockSize.x = 32 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown38_kernel"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg242, %arg243, %arg244) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %41 = memref.alloc() : memref<1x256x14x14xf16>
    %42 = memref.alloc() : memref<256xf32>
    %43 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg244, %arg151, %arg150, %41, %42, %43) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%41, %arg245) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown40_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg46, %arg246) {BlockSize.x = 32 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown41_kernel"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg245, %arg246, %arg247) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %44 = memref.alloc() : memref<1x256x14x14xf16>
    %45 = memref.alloc() : memref<256xf32>
    %46 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg247, %arg153, %arg152, %44, %45, %46) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%44, %arg242, %arg248) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown43_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg53, %arg254) {BlockSize.x = 32 : i32, GridSize.x = 4096 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown44_kernel"} : memref<512x256x1x1xf32>, memref<512x256x1x1xf16>
    byre.compute @ConvOpf16f16f16(%arg248, %arg254, %arg255) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>
    %47 = memref.alloc() : memref<1x512x7x7xf16>
    %48 = memref.alloc() : memref<512xf32>
    %49 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg255, %arg159, %arg158, %47, %48, %49) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%arg51, %arg249) {BlockSize.x = 32 : i32, GridSize.x = 36864 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown46_kernel"} : memref<512x256x3x3xf32>, memref<512x256x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg248, %arg249, %arg250) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>
    %50 = memref.alloc() : memref<1x512x7x7xf16>
    %51 = memref.alloc() : memref<512xf32>
    %52 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg250, %arg155, %arg154, %50, %51, %52) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%50, %arg251) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown48_kernel"} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>
    byre.compute @PTXOp(%arg52, %arg252) {BlockSize.x = 32 : i32, GridSize.x = 73728 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown49_kernel"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg251, %arg252, %arg253) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %53 = memref.alloc() : memref<1x512x7x7xf16>
    %54 = memref.alloc() : memref<512xf32>
    %55 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg253, %arg157, %arg156, %53, %54, %55) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%53, %47, %arg256) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown51_kernel"} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>
    byre.compute @PTXOp(%arg60, %arg257) {BlockSize.x = 32 : i32, GridSize.x = 73728 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown52_kernel"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg256, %arg257, %arg258) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %56 = memref.alloc() : memref<1x512x7x7xf16>
    %57 = memref.alloc() : memref<512xf32>
    %58 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg258, %arg161, %arg160, %56, %57, %58) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%56, %arg259) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown54_kernel"} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>
    byre.compute @PTXOp(%arg61, %arg260) {BlockSize.x = 32 : i32, GridSize.x = 73728 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown55_kernel"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    byre.compute @ConvOpf16f16f16(%arg259, %arg260, %arg261) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %59 = memref.alloc() : memref<1x512x7x7xf16>
    %60 = memref.alloc() : memref<512xf32>
    %61 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f16f32f32(%arg261, %arg163, %arg162, %59, %60, %61) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%59, %arg256, %arg262) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown57_kernel"} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>
    byre.compute @ReduceSumOpf16f16(%arg262, %1) {dimensions = dense<[3, 2]> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<1x512xf16>
    byre.compute @PTXOp(%1, %arg263) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown58_kernel"} : memref<1x512xf16>, memref<1x512xf16>
    %62 = memref.alloc() : memref<1000x512xf16>
    byre.compute @PTXOp(%arg4, %62) {BlockSize.x = 32 : i32, GridSize.x = 16000 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown59_kernel"} : memref<1000x512xf32>, memref<1000x512xf16>
    byre.compute @TransposeOpf16f16(%62, %arg264) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : memref<1000x512xf16>, memref<512x1000xf16>
    byre.compute @MatmulOpf16f16f16(%arg263, %62, %0) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<1x512xf16>, memref<1000x512xf16>, memref<1x1000xf16>
    byre.compute @PTXOp(%arg3, %arg3, %0, %arg123) {BlockSize.x = 32 : i32, GridSize.x = 32 : i32, arg_ranks = [1 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown60_kernel"} : memref<1000xf32>, memref<1000xf32>, memref<1x1000xf16>, memref<1x1000xf16>
    byre.compute @PTXOp(%3, %arg63, %arg164) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown61_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%4, %arg64, %arg165) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%6, %arg66, %arg166) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown63_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%7, %arg67, %arg167) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown64_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%9, %arg69, %arg168) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown65_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%10, %arg70, %arg169) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown66_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%12, %arg72, %arg170) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown67_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%13, %arg73, %arg171) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown68_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%15, %arg75, %arg172) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown69_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%16, %arg76, %arg173) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown70_kernel"} : memref<64xf32>, memref<64xf32>, memref<64xf32>
    byre.compute @PTXOp(%21, %arg78, %arg174) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown71_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%22, %arg79, %arg175) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%24, %arg81, %arg176) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown73_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%25, %arg82, %arg177) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown74_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%18, %arg84, %arg178) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown75_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%19, %arg85, %arg179) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown76_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%27, %arg87, %arg180) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown77_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%28, %arg88, %arg181) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown78_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%30, %arg90, %arg182) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown79_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%31, %arg91, %arg183) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown80_kernel"} : memref<128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @PTXOp(%36, %arg93, %arg184) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown81_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%37, %arg94, %arg185) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%39, %arg96, %arg186) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown83_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%40, %arg97, %arg187) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown84_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%33, %arg99, %arg188) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown85_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%34, %arg100, %arg189) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown86_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%42, %arg102, %arg190) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown87_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%43, %arg103, %arg191) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown88_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%45, %arg105, %arg192) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown89_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%46, %arg106, %arg193) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown90_kernel"} : memref<256xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @PTXOp(%51, %arg108, %arg194) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown91_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%52, %arg109, %arg195) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%54, %arg111, %arg196) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown93_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%55, %arg112, %arg197) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown94_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%48, %arg114, %arg198) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown95_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%49, %arg115, %arg199) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown96_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%57, %arg117, %arg200) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown97_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%58, %arg118, %arg201) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown98_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%60, %arg120, %arg202) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown99_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @PTXOp(%61, %arg121, %arg203) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown100_kernel"} : memref<512xf32>, memref<512xf32>, memref<512xf32>
    return
  }
}

