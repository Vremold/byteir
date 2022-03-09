// RUN: byteir-opt %s -byre-host="device-file-name=your_file" | FileCheck %s

// CHECK-LABEL: func @main
module attributes {byre.container_module, gpu.container_module} {
  gpu.module @Unknown17_kernel {
    gpu.func @Unknown17_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 4.900000e+01 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
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
        %37 = memref.load %arg1[%35, %29] : memref<1x512xf16>
        %38 = arith.divf %37, %cst : f16
        %39 = arith.cmpf ogt, %36, %cst_0 : f16
        %40 = select %39, %38, %cst_0 : f16
        memref.store %40, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown18_kernel {
    gpu.func @Unknown18_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
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
  gpu.module @Unknown22_kernel {
    gpu.func @Unknown22_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown23_kernel {
    gpu.func @Unknown23_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
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
  gpu.module @Unknown27_kernel {
    gpu.func @Unknown27_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>, %arg3: memref<1x512x7x7xf16>) kernel {
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
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown28_kernel {
    gpu.func @Unknown28_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
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
  gpu.module @Unknown32_kernel {
    gpu.func @Unknown32_kernel(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown33_kernel {
    gpu.func @Unknown33_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
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
  gpu.module @Unknown38_kernel {
    gpu.func @Unknown38_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>) kernel {
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
  gpu.module @Unknown42_kernel {
    gpu.func @Unknown42_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
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
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown43_kernel {
    gpu.func @Unknown43_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
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
  gpu.module @Unknown47_kernel {
    gpu.func @Unknown47_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown48_kernel {
    gpu.func @Unknown48_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
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
  gpu.module @Unknown52_kernel {
    gpu.func @Unknown52_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
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
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown53_kernel {
    gpu.func @Unknown53_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
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
  gpu.module @Unknown57_kernel {
    gpu.func @Unknown57_kernel(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown58_kernel {
    gpu.func @Unknown58_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
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
  gpu.module @Unknown63_kernel {
    gpu.func @Unknown63_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
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
  gpu.module @Unknown67_kernel {
    gpu.func @Unknown67_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
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
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown68_kernel {
    gpu.func @Unknown68_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
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
  gpu.module @Unknown72_kernel {
    gpu.func @Unknown72_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown73_kernel {
    gpu.func @Unknown73_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
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
  gpu.module @Unknown77_kernel {
    gpu.func @Unknown77_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
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
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown78_kernel {
    gpu.func @Unknown78_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
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
  gpu.module @Unknown82_kernel {
    gpu.func @Unknown82_kernel(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown83_kernel {
    gpu.func @Unknown83_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
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
  gpu.module @Unknown88_kernel {
    gpu.func @Unknown88_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) kernel {
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
  gpu.module @Unknown92_kernel {
    gpu.func @Unknown92_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
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
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown93_kernel {
    gpu.func @Unknown93_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
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
  gpu.module @Unknown97_kernel {
    gpu.func @Unknown97_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown98_kernel {
    gpu.func @Unknown98_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
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
  gpu.module @Unknown102_kernel {
    gpu.func @Unknown102_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
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
        %38 = memref.load %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
        %39 = arith.addf %37, %38 : f16
        %40 = arith.cmpf ogt, %36, %cst : f16
        %41 = select %40, %39, %cst : f16
        memref.store %41, %arg3[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown103_kernel {
    gpu.func @Unknown103_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
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
  gpu.module @Unknown107_kernel {
    gpu.func @Unknown107_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
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
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown108_kernel {
    gpu.func @Unknown108_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
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
  gpu.module @Unknown112_kernel {
    gpu.func @Unknown112_kernel(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c200704 = arith.constant 200704 : index
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
        memref.store %38, %arg2[%35, %29, %19, %9] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown113_kernel {
    gpu.func @Unknown113_kernel(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>, %arg2: memref<1x64x112x112xf16>) kernel {
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
        %37 = memref.load %arg1[%35, %29, %19, %9] : memref<1x64x112x112xf16>
        %38 = arith.cmpf ogt, %36, %cst : f16
        %39 = select %38, %37, %cst : f16
        memref.store %39, %arg2[%35, %29, %19, %9] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
  }
  gpu.module @Unknown114_kernel {
    gpu.func @Unknown114_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) kernel {
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
  gpu.module @Unknown117_kernel {
    gpu.func @Unknown117_kernel(%arg0: memref<64x3x7x7xf16>, %arg1: memref<64x3x7x7xf32>) kernel {
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
  gpu.module @Unknown118_kernel {
    gpu.func @Unknown118_kernel(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf32>) kernel {
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
        %16 = memref.load %arg0[%15, %9] : memref<1x1000xf16>
        %17 = arith.extf %16 : f16 to f32
        memref.store %17, %arg1[%15, %9] : memref<1x1000xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown119_kernel {
    gpu.func @Unknown119_kernel(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) kernel {
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
  gpu.module @Unknown120_kernel {
    gpu.func @Unknown120_kernel(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
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
  gpu.module @Unknown121_kernel {
    gpu.func @Unknown121_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
  gpu.module @Unknown122_kernel {
    gpu.func @Unknown122_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
  gpu.module @Unknown123_kernel {
    gpu.func @Unknown123_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
  gpu.module @Unknown124_kernel {
    gpu.func @Unknown124_kernel(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
  gpu.module @Unknown125_kernel {
    gpu.func @Unknown125_kernel(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
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
  gpu.module @Unknown126_kernel {
    gpu.func @Unknown126_kernel(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
  gpu.module @Unknown127_kernel {
    gpu.func @Unknown127_kernel(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
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
  gpu.module @Unknown128_kernel {
    gpu.func @Unknown128_kernel(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
  gpu.module @Unknown129_kernel {
    gpu.func @Unknown129_kernel(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
  gpu.module @Unknown130_kernel {
    gpu.func @Unknown130_kernel(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
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
  gpu.module @Unknown131_kernel {
    gpu.func @Unknown131_kernel(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
  gpu.module @Unknown132_kernel {
    gpu.func @Unknown132_kernel(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
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
  gpu.module @Unknown133_kernel {
    gpu.func @Unknown133_kernel(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
  gpu.module @Unknown134_kernel {
    gpu.func @Unknown134_kernel(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
  gpu.module @Unknown135_kernel {
    gpu.func @Unknown135_kernel(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
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
  gpu.module @Unknown136_kernel {
    gpu.func @Unknown136_kernel(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
  gpu.module @Unknown137_kernel {
    gpu.func @Unknown137_kernel(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
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
  gpu.module @Unknown138_kernel {
    gpu.func @Unknown138_kernel(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
  gpu.module @Unknown139_kernel {
    gpu.func @Unknown139_kernel(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
  func @main(%arg0: memref<64xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<64xf32> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<64xf32> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<64xf32> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<64xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<64xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<64xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<64xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<64xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<64xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<256xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<256xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<256xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<256xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<256xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<256xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<256xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<256xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<256xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<256xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<512xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<512xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<512xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<512xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<512xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<512xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<512xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<64xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<64xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<64xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<64xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<64xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<64xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<64xf32> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<64xf32> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<64xf32> {byre.argname = "Input48", byre.argtype = 1 : i32}, %arg49: memref<64xf32> {byre.argname = "Input49", byre.argtype = 1 : i32}, %arg50: memref<128xf32> {byre.argname = "Input50", byre.argtype = 1 : i32}, %arg51: memref<128xf32> {byre.argname = "Input51", byre.argtype = 1 : i32}, %arg52: memref<128xf32> {byre.argname = "Input52", byre.argtype = 1 : i32}, %arg53: memref<128xf32> {byre.argname = "Input53", byre.argtype = 1 : i32}, %arg54: memref<128xf32> {byre.argname = "Input54", byre.argtype = 1 : i32}, %arg55: memref<128xf32> {byre.argname = "Input55", byre.argtype = 1 : i32}, %arg56: memref<128xf32> {byre.argname = "Input56", byre.argtype = 1 : i32}, %arg57: memref<128xf32> {byre.argname = "Input57", byre.argtype = 1 : i32}, %arg58: memref<128xf32> {byre.argname = "Input58", byre.argtype = 1 : i32}, %arg59: memref<128xf32> {byre.argname = "Input59", byre.argtype = 1 : i32}, %arg60: memref<256xf32> {byre.argname = "Input60", byre.argtype = 1 : i32}, %arg61: memref<256xf32> {byre.argname = "Input61", byre.argtype = 1 : i32}, %arg62: memref<256xf32> {byre.argname = "Input62", byre.argtype = 1 : i32}, %arg63: memref<256xf32> {byre.argname = "Input63", byre.argtype = 1 : i32}, %arg64: memref<256xf32> {byre.argname = "Input64", byre.argtype = 1 : i32}, %arg65: memref<256xf32> {byre.argname = "Input65", byre.argtype = 1 : i32}, %arg66: memref<256xf32> {byre.argname = "Input66", byre.argtype = 1 : i32}, %arg67: memref<256xf32> {byre.argname = "Input67", byre.argtype = 1 : i32}, %arg68: memref<256xf32> {byre.argname = "Input68", byre.argtype = 1 : i32}, %arg69: memref<256xf32> {byre.argname = "Input69", byre.argtype = 1 : i32}, %arg70: memref<512xf32> {byre.argname = "Input70", byre.argtype = 1 : i32}, %arg71: memref<512xf32> {byre.argname = "Input71", byre.argtype = 1 : i32}, %arg72: memref<512xf32> {byre.argname = "Input72", byre.argtype = 1 : i32}, %arg73: memref<512xf32> {byre.argname = "Input73", byre.argtype = 1 : i32}, %arg74: memref<512xf32> {byre.argname = "Input74", byre.argtype = 1 : i32}, %arg75: memref<512xf32> {byre.argname = "Input75", byre.argtype = 1 : i32}, %arg76: memref<512xf32> {byre.argname = "Input76", byre.argtype = 1 : i32}, %arg77: memref<512xf32> {byre.argname = "Input77", byre.argtype = 1 : i32}, %arg78: memref<512xf32> {byre.argname = "Input78", byre.argtype = 1 : i32}, %arg79: memref<512xf32> {byre.argname = "Input79", byre.argtype = 1 : i32}, %arg80: memref<64x3x7x7xf16> {byre.argname = "Input80", byre.argtype = 1 : i32}, %arg81: memref<1x3x224x224xf16> {byre.argname = "Input81", byre.argtype = 1 : i32}, %arg82: memref<1x64x112x112xf16> {byre.argname = "Input82", byre.argtype = 1 : i32}, %arg83: memref<1x64x112x112xf16> {byre.argname = "Input83", byre.argtype = 1 : i32}, %arg84: memref<1x64x56x56xf16> {byre.argname = "Input84", byre.argtype = 1 : i32}, %arg85: memref<64x64x3x3xf16> {byre.argname = "Input85", byre.argtype = 1 : i32}, %arg86: memref<1x64x56x56xf16> {byre.argname = "Input86", byre.argtype = 1 : i32}, %arg87: memref<1x64x56x56xf16> {byre.argname = "Input87", byre.argtype = 1 : i32}, %arg88: memref<64x64x3x3xf16> {byre.argname = "Input88", byre.argtype = 1 : i32}, %arg89: memref<1x64x56x56xf16> {byre.argname = "Input89", byre.argtype = 1 : i32}, %arg90: memref<1x64x56x56xf16> {byre.argname = "Input90", byre.argtype = 1 : i32}, %arg91: memref<64x64x3x3xf16> {byre.argname = "Input91", byre.argtype = 1 : i32}, %arg92: memref<1x64x56x56xf16> {byre.argname = "Input92", byre.argtype = 1 : i32}, %arg93: memref<1x64x56x56xf16> {byre.argname = "Input93", byre.argtype = 1 : i32}, %arg94: memref<64x64x3x3xf16> {byre.argname = "Input94", byre.argtype = 1 : i32}, %arg95: memref<1x64x56x56xf16> {byre.argname = "Input95", byre.argtype = 1 : i32}, %arg96: memref<1x64x56x56xf16> {byre.argname = "Input96", byre.argtype = 1 : i32}, %arg97: memref<128x64x3x3xf16> {byre.argname = "Input97", byre.argtype = 1 : i32}, %arg98: memref<1x128x28x28xf16> {byre.argname = "Input98", byre.argtype = 1 : i32}, %arg99: memref<1x128x28x28xf16> {byre.argname = "Input99", byre.argtype = 1 : i32}, %arg100: memref<128x128x3x3xf16> {byre.argname = "Input100", byre.argtype = 1 : i32}, %arg101: memref<1x128x28x28xf16> {byre.argname = "Input101", byre.argtype = 1 : i32}, %arg102: memref<128x64x1x1xf16> {byre.argname = "Input102", byre.argtype = 1 : i32}, %arg103: memref<1x128x28x28xf16> {byre.argname = "Input103", byre.argtype = 1 : i32}, %arg104: memref<1x128x28x28xf16> {byre.argname = "Input104", byre.argtype = 1 : i32}, %arg105: memref<128x128x3x3xf16> {byre.argname = "Input105", byre.argtype = 1 : i32}, %arg106: memref<1x128x28x28xf16> {byre.argname = "Input106", byre.argtype = 1 : i32}, %arg107: memref<1x128x28x28xf16> {byre.argname = "Input107", byre.argtype = 1 : i32}, %arg108: memref<128x128x3x3xf16> {byre.argname = "Input108", byre.argtype = 1 : i32}, %arg109: memref<1x128x28x28xf16> {byre.argname = "Input109", byre.argtype = 1 : i32}, %arg110: memref<1x128x28x28xf16> {byre.argname = "Input110", byre.argtype = 1 : i32}, %arg111: memref<256x128x3x3xf16> {byre.argname = "Input111", byre.argtype = 1 : i32}, %arg112: memref<1x256x14x14xf16> {byre.argname = "Input112", byre.argtype = 1 : i32}, %arg113: memref<1x256x14x14xf16> {byre.argname = "Input113", byre.argtype = 1 : i32}, %arg114: memref<256x256x3x3xf16> {byre.argname = "Input114", byre.argtype = 1 : i32}, %arg115: memref<1x256x14x14xf16> {byre.argname = "Input115", byre.argtype = 1 : i32}, %arg116: memref<256x128x1x1xf16> {byre.argname = "Input116", byre.argtype = 1 : i32}, %arg117: memref<1x256x14x14xf16> {byre.argname = "Input117", byre.argtype = 1 : i32}, %arg118: memref<1x256x14x14xf16> {byre.argname = "Input118", byre.argtype = 1 : i32}, %arg119: memref<256x256x3x3xf16> {byre.argname = "Input119", byre.argtype = 1 : i32}, %arg120: memref<1x256x14x14xf16> {byre.argname = "Input120", byre.argtype = 1 : i32}, %arg121: memref<1x256x14x14xf16> {byre.argname = "Input121", byre.argtype = 1 : i32}, %arg122: memref<256x256x3x3xf16> {byre.argname = "Input122", byre.argtype = 1 : i32}, %arg123: memref<1x256x14x14xf16> {byre.argname = "Input123", byre.argtype = 1 : i32}, %arg124: memref<1x256x14x14xf16> {byre.argname = "Input124", byre.argtype = 1 : i32}, %arg125: memref<512x256x3x3xf16> {byre.argname = "Input125", byre.argtype = 1 : i32}, %arg126: memref<1x512x7x7xf16> {byre.argname = "Input126", byre.argtype = 1 : i32}, %arg127: memref<1x512x7x7xf16> {byre.argname = "Input127", byre.argtype = 1 : i32}, %arg128: memref<512x512x3x3xf16> {byre.argname = "Input128", byre.argtype = 1 : i32}, %arg129: memref<1x512x7x7xf16> {byre.argname = "Input129", byre.argtype = 1 : i32}, %arg130: memref<512x256x1x1xf16> {byre.argname = "Input130", byre.argtype = 1 : i32}, %arg131: memref<1x512x7x7xf16> {byre.argname = "Input131", byre.argtype = 1 : i32}, %arg132: memref<1x512x7x7xf16> {byre.argname = "Input132", byre.argtype = 1 : i32}, %arg133: memref<512x512x3x3xf16> {byre.argname = "Input133", byre.argtype = 1 : i32}, %arg134: memref<1x512x7x7xf16> {byre.argname = "Input134", byre.argtype = 1 : i32}, %arg135: memref<1x512x7x7xf16> {byre.argname = "Input135", byre.argtype = 1 : i32}, %arg136: memref<512x512x3x3xf16> {byre.argname = "Input136", byre.argtype = 1 : i32}, %arg137: memref<1x512x7x7xf16> {byre.argname = "Input137", byre.argtype = 1 : i32}, %arg138: memref<1x512x7x7xf16> {byre.argname = "Input138", byre.argtype = 1 : i32}, %arg139: memref<1x512xf16> {byre.argname = "Input139", byre.argtype = 1 : i32}, %arg140: memref<512x1000xf16> {byre.argname = "Input140", byre.argtype = 1 : i32}, %arg141: memref<1x1000xf16> {byre.argname = "Input141", byre.argtype = 1 : i32}, %arg142: memref<64xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg143: memref<64xf32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg144: memref<64x3x7x7xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg145: memref<1000xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg146: memref<1000x512xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg147: memref<64xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg148: memref<64xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg149: memref<64xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg150: memref<64xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg151: memref<64x64x3x3xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg152: memref<64x64x3x3xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg153: memref<64xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg154: memref<64xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg155: memref<64xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg156: memref<64xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg157: memref<64x64x3x3xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg158: memref<64x64x3x3xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg159: memref<128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg160: memref<128xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg161: memref<128xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg162: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg163: memref<128x64x3x3xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg164: memref<128x128x3x3xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg165: memref<128x64x1x1xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg166: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg167: memref<128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg168: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg169: memref<128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg170: memref<128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg171: memref<128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg172: memref<128x128x3x3xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg173: memref<128x128x3x3xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg174: memref<256xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg175: memref<256xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg176: memref<256xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg177: memref<256xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg178: memref<256x128x3x3xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg179: memref<256x256x3x3xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg180: memref<256x128x1x1xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg181: memref<256xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg182: memref<256xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg183: memref<256xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg184: memref<256xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg185: memref<256xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}, %arg186: memref<256xf32> {byre.argname = "Output44", byre.argtype = 2 : i32}, %arg187: memref<256x256x3x3xf32> {byre.argname = "Output45", byre.argtype = 2 : i32}, %arg188: memref<256x256x3x3xf32> {byre.argname = "Output46", byre.argtype = 2 : i32}, %arg189: memref<512xf32> {byre.argname = "Output47", byre.argtype = 2 : i32}, %arg190: memref<512xf32> {byre.argname = "Output48", byre.argtype = 2 : i32}, %arg191: memref<512xf32> {byre.argname = "Output49", byre.argtype = 2 : i32}, %arg192: memref<512xf32> {byre.argname = "Output50", byre.argtype = 2 : i32}, %arg193: memref<512x256x3x3xf32> {byre.argname = "Output51", byre.argtype = 2 : i32}, %arg194: memref<512x512x3x3xf32> {byre.argname = "Output52", byre.argtype = 2 : i32}, %arg195: memref<512x256x1x1xf32> {byre.argname = "Output53", byre.argtype = 2 : i32}, %arg196: memref<512xf32> {byre.argname = "Output54", byre.argtype = 2 : i32}, %arg197: memref<512xf32> {byre.argname = "Output55", byre.argtype = 2 : i32}, %arg198: memref<512xf32> {byre.argname = "Output56", byre.argtype = 2 : i32}, %arg199: memref<512xf32> {byre.argname = "Output57", byre.argtype = 2 : i32}, %arg200: memref<512xf32> {byre.argname = "Output58", byre.argtype = 2 : i32}, %arg201: memref<512xf32> {byre.argname = "Output59", byre.argtype = 2 : i32}, %arg202: memref<512x512x3x3xf32> {byre.argname = "Output60", byre.argtype = 2 : i32}, %arg203: memref<512x512x3x3xf32> {byre.argname = "Output61", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<1000x512xf16>
    %1 = memref.alloc() : memref<1000xf32>
    %2 = memref.alloc() : memref<1x64x112x112xf16>
    %3 = memref.alloc() : memref<1x512xf16>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg82, %arg1, %arg0, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>
    %6 = memref.alloc() : memref<64xf32>
    %7 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg86, %arg3, %arg2, %6, %7) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>
    %8 = memref.alloc() : memref<64xf32>
    %9 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg89, %arg5, %arg4, %8, %9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>
    %10 = memref.alloc() : memref<64xf32>
    %11 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg92, %arg7, %arg6, %10, %11) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>
    %12 = memref.alloc() : memref<64xf32>
    %13 = memref.alloc() : memref<64xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg95, %arg9, %arg8, %12, %13) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>
    %14 = memref.alloc() : memref<128xf32>
    %15 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg98, %arg11, %arg10, %14, %15) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>
    %16 = memref.alloc() : memref<128xf32>
    %17 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg101, %arg13, %arg12, %16, %17) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>
    %18 = memref.alloc() : memref<128xf32>
    %19 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg106, %arg17, %arg16, %18, %19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>
    %20 = memref.alloc() : memref<128xf32>
    %21 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg109, %arg19, %arg18, %20, %21) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>
    %22 = memref.alloc() : memref<256xf32>
    %23 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg112, %arg21, %arg20, %22, %23) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>
    %24 = memref.alloc() : memref<256xf32>
    %25 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg115, %arg23, %arg22, %24, %25) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>
    %26 = memref.alloc() : memref<256xf32>
    %27 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg120, %arg27, %arg26, %26, %27) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>
    %28 = memref.alloc() : memref<256xf32>
    %29 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg123, %arg29, %arg28, %28, %29) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>
    %30 = memref.alloc() : memref<512xf32>
    %31 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg126, %arg31, %arg30, %30, %31) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>
    %32 = memref.alloc() : memref<512xf32>
    %33 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg129, %arg33, %arg32, %32, %33) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>
    %34 = memref.alloc() : memref<512xf32>
    %35 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg134, %arg37, %arg36, %34, %35) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>
    %36 = memref.alloc() : memref<512xf32>
    %37 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg137, %arg39, %arg38, %36, %37) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>
    byre.compute @MatmulOpf16f16f16(%arg141, %arg140, %3) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<1x1000xf16>, memref<512x1000xf16>, memref<1x512xf16>
    %38 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @PTXOp(%arg138, %3, %38) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 2 : i32, 4 : i32], kernel_name = "Unknown17_kernel"} : memref<1x512x7x7xf16>, memref<1x512xf16>, memref<1x512x7x7xf16>
    %39 = memref.alloc() : memref<512xf32>
    byre.compute @PTXOp(%37, %39) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown18_kernel"} : memref<512xf32>, memref<512xf32>
    %40 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg137, %arg39, %36, %39, %38, %40, %arg201, %arg200) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %41 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%40, %arg136, %41) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %42 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg135, %40, %42) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>
    %43 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @PTXOp(%arg135, %41, %43) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown22_kernel"} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>
    %44 = memref.alloc() : memref<512xf32>
    byre.compute @PTXOp(%35, %44) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown23_kernel"} : memref<512xf32>, memref<512xf32>
    %45 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg134, %arg37, %34, %44, %43, %45, %arg199, %arg198) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %46 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%45, %arg133, %46) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %47 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg132, %45, %47) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>
    %48 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @PTXOp(%arg132, %38, %46, %48) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown27_kernel"} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>
    %49 = memref.alloc() : memref<512xf32>
    byre.compute @PTXOp(%33, %49) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown28_kernel"} : memref<512xf32>, memref<512xf32>
    %50 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg129, %arg33, %32, %49, %48, %50, %arg192, %arg191) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %51 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%50, %arg128, %51) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>
    %52 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg127, %50, %52) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>
    %53 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @PTXOp(%arg127, %51, %53) {BlockSize.x = 32 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown32_kernel"} : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>
    %54 = memref.alloc() : memref<512xf32>
    byre.compute @PTXOp(%31, %54) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown33_kernel"} : memref<512xf32>, memref<512xf32>
    %55 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg126, %arg31, %30, %54, %53, %55, %arg190, %arg189) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %56 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%55, %arg125, %56) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x256x3x3xf16>, memref<1x256x14x14xf16>
    %57 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg124, %55, %57) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<512x256x3x3xf16>
    %58 = memref.alloc() : memref<512xf32>
    %59 = memref.alloc() : memref<512xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg131, %arg35, %arg34, %58, %59) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>
    %60 = memref.alloc() : memref<512xf32>
    byre.compute @PTXOp(%59, %60) {BlockSize.x = 32 : i32, GridSize.x = 16 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown38_kernel"} : memref<512xf32>, memref<512xf32>
    %61 = memref.alloc() : memref<1x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg131, %arg35, %58, %60, %48, %61, %arg197, %arg196) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %62 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%61, %arg130, %62) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x256x14x14xf16>
    %63 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg124, %61, %63) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>
    %64 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg124, %62, %56, %64) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown42_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    %65 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%29, %65) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown43_kernel"} : memref<256xf32>, memref<256xf32>
    %66 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg123, %arg29, %28, %65, %64, %66, %arg186, %arg185) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %67 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%66, %arg122, %67) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %68 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg121, %66, %68) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>
    %69 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg121, %67, %69) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown47_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    %70 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%27, %70) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown48_kernel"} : memref<256xf32>, memref<256xf32>
    %71 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg120, %arg27, %26, %70, %69, %71, %arg184, %arg183) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %72 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%71, %arg119, %72) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %73 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg118, %71, %73) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>
    %74 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg118, %64, %72, %74) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown52_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    %75 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%25, %75) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown53_kernel"} : memref<256xf32>, memref<256xf32>
    %76 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg115, %arg23, %24, %75, %74, %76, %arg177, %arg176) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %77 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%76, %arg114, %77) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>
    %78 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg113, %76, %78) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>
    %79 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @PTXOp(%arg113, %77, %79) {BlockSize.x = 32 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown57_kernel"} : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>
    %80 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%23, %80) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown58_kernel"} : memref<256xf32>, memref<256xf32>
    %81 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg112, %arg21, %22, %80, %79, %81, %arg175, %arg174) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %82 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%81, %arg111, %82) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x128x3x3xf16>, memref<1x128x28x28xf16>
    %83 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg110, %81, %83) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<256x128x3x3xf16>
    %84 = memref.alloc() : memref<256xf32>
    %85 = memref.alloc() : memref<256xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg117, %arg25, %arg24, %84, %85) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>
    %86 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%85, %86) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown63_kernel"} : memref<256xf32>, memref<256xf32>
    %87 = memref.alloc() : memref<1x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg117, %arg25, %84, %86, %74, %87, %arg182, %arg181) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %88 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%87, %arg116, %88) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x128x28x28xf16>
    %89 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg110, %87, %89) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>
    %90 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg110, %88, %82, %90) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown67_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    %91 = memref.alloc() : memref<128xf32>
    byre.compute @PTXOp(%21, %91) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown68_kernel"} : memref<128xf32>, memref<128xf32>
    %92 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg109, %arg19, %20, %91, %90, %92, %arg171, %arg170) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %93 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%92, %arg108, %93) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %94 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg107, %92, %94) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>
    %95 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg107, %93, %95) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown72_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    %96 = memref.alloc() : memref<128xf32>
    byre.compute @PTXOp(%19, %96) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown73_kernel"} : memref<128xf32>, memref<128xf32>
    %97 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg106, %arg17, %18, %96, %95, %97, %arg169, %arg168) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %98 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%97, %arg105, %98) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %99 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg104, %97, %99) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>
    %100 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg104, %90, %98, %100) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown77_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    %101 = memref.alloc() : memref<128xf32>
    byre.compute @PTXOp(%17, %101) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown78_kernel"} : memref<128xf32>, memref<128xf32>
    %102 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg101, %arg13, %16, %101, %100, %102, %arg162, %arg161) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %103 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%102, %arg100, %103) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>
    %104 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg99, %102, %104) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>
    %105 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @PTXOp(%arg99, %103, %105) {BlockSize.x = 32 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown82_kernel"} : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>
    %106 = memref.alloc() : memref<128xf32>
    byre.compute @PTXOp(%15, %106) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown83_kernel"} : memref<128xf32>, memref<128xf32>
    %107 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg98, %arg11, %14, %106, %105, %107, %arg160, %arg159) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %108 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%107, %arg97, %108) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x64x3x3xf16>, memref<1x64x56x56xf16>
    %109 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg96, %107, %109) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<128x64x3x3xf16>
    %110 = memref.alloc() : memref<128xf32>
    %111 = memref.alloc() : memref<128xf32>
    byre.compute @BatchNormTrainingOpf16f32f32f32f32(%arg103, %arg15, %arg14, %110, %111) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>
    %112 = memref.alloc() : memref<128xf32>
    byre.compute @PTXOp(%111, %112) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown88_kernel"} : memref<128xf32>, memref<128xf32>
    %113 = memref.alloc() : memref<1x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg103, %arg15, %110, %112, %100, %113, %arg167, %arg166) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %114 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%113, %arg102, %114) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x64x56x56xf16>
    %115 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg96, %113, %115) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>
    %116 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg96, %114, %108, %116) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown92_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    %117 = memref.alloc() : memref<64xf32>
    byre.compute @PTXOp(%13, %117) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown93_kernel"} : memref<64xf32>, memref<64xf32>
    %118 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg95, %arg9, %12, %117, %116, %118, %arg156, %arg155) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %119 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%118, %arg94, %119) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %120 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg93, %118, %120) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %121 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg93, %119, %121) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown97_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    %122 = memref.alloc() : memref<64xf32>
    byre.compute @PTXOp(%11, %122) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown98_kernel"} : memref<64xf32>, memref<64xf32>
    %123 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg92, %arg7, %10, %122, %121, %123, %arg154, %arg153) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %124 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%123, %arg91, %124) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %125 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg90, %123, %125) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %126 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg90, %116, %124, %126) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown102_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    %127 = memref.alloc() : memref<64xf32>
    byre.compute @PTXOp(%9, %127) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown103_kernel"} : memref<64xf32>, memref<64xf32>
    %128 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg89, %arg5, %8, %127, %126, %128, %arg150, %arg149) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %129 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%128, %arg88, %129) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %130 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg87, %128, %130) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %131 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @PTXOp(%arg87, %129, %131) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown107_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    %132 = memref.alloc() : memref<64xf32>
    byre.compute @PTXOp(%7, %132) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown108_kernel"} : memref<64xf32>, memref<64xf32>
    %133 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg86, %arg3, %6, %132, %131, %133, %arg148, %arg147) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %134 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%133, %arg85, %134) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>
    %135 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg84, %133, %135) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>
    %136 = memref.alloc() : memref<1x64x56x56xf16>
    byre.compute @PTXOp(%126, %134, %136) {BlockSize.x = 32 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown112_kernel"} : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>
    byre.compute @PoolMaxGradOpf16f16f16(%arg83, %136, %2) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<1x64x112x112xf16>
    %137 = memref.alloc() : memref<1x64x112x112xf16>
    byre.compute @PTXOp(%arg83, %2, %137) {BlockSize.x = 32 : i32, GridSize.x = 25088 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown113_kernel"} : memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>
    %138 = memref.alloc() : memref<64xf32>
    byre.compute @PTXOp(%5, %138) {BlockSize.x = 32 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown114_kernel"} : memref<64xf32>, memref<64xf32>
    %139 = memref.alloc() : memref<1x64x112x112xf16>
    byre.compute @BatchNormGradOpf16f32f32f32f16f16f32f32(%arg82, %arg1, %4, %138, %137, %139, %arg143, %arg142) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %140 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%arg81, %139, %140) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<64x3x7x7xf16>
    byre.compute @PTXOp(%140, %arg144) {BlockSize.x = 32 : i32, GridSize.x = 294 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown117_kernel"} : memref<64x3x7x7xf16>, memref<64x3x7x7xf32>
    %141 = memref.alloc() : memref<1x1000xf32>
    byre.compute @PTXOp(%arg141, %141) {BlockSize.x = 32 : i32, GridSize.x = 32 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown118_kernel"} : memref<1x1000xf16>, memref<1x1000xf32>
    byre.compute @ReduceSumOpf32f32(%141, %1) {dimensions = dense<0> : tensor<1xi64>} : memref<1x1000xf32>, memref<1000xf32>
    byre.compute @PTXOp(%1, %arg145) {BlockSize.x = 32 : i32, GridSize.x = 32 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown119_kernel"} : memref<1000xf32>, memref<1000xf32>
    byre.compute @MatmulOpf16f16f16(%arg141, %arg139, %0) {lhs_contracting_dimension = 0 : i64, rhs_contracting_dimension = 0 : i64} : memref<1x1000xf16>, memref<1x512xf16>, memref<1000x512xf16>
    byre.compute @PTXOp(%0, %arg146) {BlockSize.x = 32 : i32, GridSize.x = 16000 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown120_kernel"} : memref<1000x512xf16>, memref<1000x512xf32>
    byre.compute @PTXOp(%135, %arg151) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown121_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%130, %arg152) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown122_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%125, %arg157) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown123_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%120, %arg158) {BlockSize.x = 32 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown124_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%109, %arg163) {BlockSize.x = 32 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown125_kernel"} : memref<128x64x3x3xf16>, memref<128x64x3x3xf32>
    byre.compute @PTXOp(%104, %arg164) {BlockSize.x = 32 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown126_kernel"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%115, %arg165) {BlockSize.x = 32 : i32, GridSize.x = 256 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown127_kernel"} : memref<128x64x1x1xf16>, memref<128x64x1x1xf32>
    byre.compute @PTXOp(%99, %arg172) {BlockSize.x = 32 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown128_kernel"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%94, %arg173) {BlockSize.x = 32 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown129_kernel"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%83, %arg178) {BlockSize.x = 32 : i32, GridSize.x = 9216 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown130_kernel"} : memref<256x128x3x3xf16>, memref<256x128x3x3xf32>
    byre.compute @PTXOp(%78, %arg179) {BlockSize.x = 32 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown131_kernel"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%89, %arg180) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown132_kernel"} : memref<256x128x1x1xf16>, memref<256x128x1x1xf32>
    byre.compute @PTXOp(%73, %arg187) {BlockSize.x = 32 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown133_kernel"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%68, %arg188) {BlockSize.x = 32 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown134_kernel"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%57, %arg193) {BlockSize.x = 32 : i32, GridSize.x = 36864 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown135_kernel"} : memref<512x256x3x3xf16>, memref<512x256x3x3xf32>
    byre.compute @PTXOp(%52, %arg194) {BlockSize.x = 32 : i32, GridSize.x = 73728 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown136_kernel"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    byre.compute @PTXOp(%63, %arg195) {BlockSize.x = 32 : i32, GridSize.x = 4096 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown137_kernel"} : memref<512x256x1x1xf16>, memref<512x256x1x1xf32>
    byre.compute @PTXOp(%47, %arg202) {BlockSize.x = 32 : i32, GridSize.x = 73728 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown138_kernel"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    byre.compute @PTXOp(%42, %arg203) {BlockSize.x = 32 : i32, GridSize.x = 73728 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown139_kernel"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    return
  }
}

