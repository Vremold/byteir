// RUN: byteir-opt %s -rewrite-affine-to-memref -coalesced-for-to-gpu -cse -sccp -cse -lower-affine -gpu-kernel-outlining -cse | FileCheck %s

// CHECK-LABEL: func @main

#map0 = affine_map<(d0) -> (d0 mod 128)>
#map1 = affine_map<(d0) -> (d0 floordiv 128)>
#map2 = affine_map<(d0) -> ((d0 floordiv 128) mod 128)>
#map3 = affine_map<(d0) -> ((d0 floordiv 128) floordiv 128)>
module  {
  func private @Unknown0(%arg0: memref<2x128xi64>, %arg1: memref<256xi64>, %arg2: memref<256xi64>, %arg3: memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
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
    affine.for %arg4 = 0 to 256 {
      %10 = affine.apply #map0(%arg4)
      %11 = affine.apply #map1(%arg4)
      %12 = affine.load %arg0[%11, %10] : memref<2x128xi64>
      %13 = trunci %12 : i64 to i32
      %14 = builtin.unrealized_conversion_cast %13 : i32 to ui32
      affine.store %14, %0[%11, %10] : memref<2x128xui32>
      %15 = affine.load %8[%11, %10] : memref<2x128xi64>
      %16 = affine.load %7[%11, %10] : memref<2x128xi64>
      %17 = addi %12, %16 : i64
      %18 = cmpi slt, %12, %15 : i64
      %19 = select %18, %17, %12 : i64
      affine.store %19, %4[%11, %10] : memref<2x128xi64>
      %20 = affine.load %3[%11, %10] : memref<2x128xf64>
      %21 = sitofp %12 : i64 to f64
      %22 = cmpf une, %21, %20 : f64
      affine.store %22, %1[%11, %10] : memref<2x128xi1>
    }
    return %9, %6, %2 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown1(%arg0: memref<128xi64>, %arg1: memref<256xi64>, %arg2: memref<256xi64>, %arg3: memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
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
    affine.for %arg4 = 0 to 256 {
      %10 = affine.apply #map0(%arg4)
      %11 = affine.apply #map1(%arg4)
      %12 = affine.load %arg0[%10] : memref<128xi64>
      %13 = trunci %12 : i64 to i32
      %14 = builtin.unrealized_conversion_cast %13 : i32 to ui32
      affine.store %14, %0[%11, %10] : memref<2x128xui32>
      %15 = affine.load %8[%11, %10] : memref<2x128xi64>
      %16 = affine.load %7[%11, %10] : memref<2x128xi64>
      %17 = addi %12, %16 : i64
      %18 = cmpi slt, %12, %15 : i64
      %19 = select %18, %17, %12 : i64
      affine.store %19, %4[%11, %10] : memref<2x128xi64>
      %20 = affine.load %3[%11, %10] : memref<2x128xf64>
      %21 = sitofp %12 : i64 to f64
      %22 = cmpf une, %21, %20 : f64
      affine.store %22, %1[%11, %10] : memref<2x128xi1>
    }
    return %9, %6, %2 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown2(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %1 = memref.alloc() : memref<2x128x128xf32>
    %2 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    affine.for %arg2 = 0 to 32768 {
      %3 = affine.apply #map0(%arg2)
      %4 = affine.apply #map2(%arg2)
      %5 = affine.apply #map3(%arg2)
      %6 = affine.load %0[%5, %4, %3] : memref<2x128x128xf32>
      %7 = affine.load %2[%5, %4, %3] : memref<2x128x128xf32>
      %8 = addf %6, %7 : f32
      affine.store %8, %1[%5, %4, %3] : memref<2x128x128xf32>
    }
    return %1 : memref<2x128x128xf32>
  }
  func private @Unknown3(%arg0: memref<1x128xi64>, %arg1: memref<128xi64>, %arg2: memref<128xi64>, %arg3: memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    %1 = memref.alloc() : memref<128xi1>
    %2 = memref.alloc() : memref<128xi64>
    %3 = memref.expand_shape %2 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    %4 = memref.alloc() : memref<128xui32>
    affine.for %arg4 = 0 to 128 {
      %5 = affine.load %0[%arg4] : memref<128xi64>
      %6 = trunci %5 : i64 to i32
      %7 = builtin.unrealized_conversion_cast %6 : i32 to ui32
      affine.store %7, %4[%arg4] : memref<128xui32>
      %8 = affine.load %arg1[%arg4] : memref<128xi64>
      %9 = affine.load %arg2[%arg4] : memref<128xi64>
      %10 = addi %5, %9 : i64
      %11 = cmpi slt, %5, %8 : i64
      %12 = select %11, %10, %5 : i64
      affine.store %12, %2[%arg4] : memref<128xi64>
      %13 = affine.load %arg3[%arg4] : memref<128xf64>
      %14 = sitofp %5 : i64 to f64
      %15 = cmpf une, %14, %13 : f64
      affine.store %15, %1[%arg4] : memref<128xi1>
    }
    return %4, %3, %1 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func private @Unknown4(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    affine.for %arg4 = 0 to 32768 {
      %1 = affine.apply #map0(%arg4)
      %2 = affine.apply #map2(%arg4)
      %3 = affine.apply #map3(%arg4)
      %4 = affine.load %arg0[%3, %2, %1] : memref<2x128x128xf32>
      %5 = affine.load %arg1[%3, %2, %1] : memref<2x128x128xf32>
      %6 = affine.load %arg2[%3, %2, %1] : memref<2x128x128xf32>
      %7 = affine.load %arg3[%3, %2, %1] : memref<2x128x128xf32>
      %8 = addf %4, %5 : f32
      %9 = addf %8, %6 : f32
      %10 = addf %9, %7 : f32
      affine.store %10, %0[%3, %2, %1] : memref<2x128x128xf32>
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown5(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    affine.for %arg4 = 0 to 32768 {
      %1 = affine.apply #map0(%arg4)
      %2 = affine.apply #map2(%arg4)
      %3 = affine.apply #map3(%arg4)
      %4 = affine.load %arg0[%3, %2, %1] : memref<2x128x128xf32>
      %5 = affine.load %arg1[%3, %2, %1] : memref<2x128x128xf32>
      %6 = affine.load %arg2[%3, %2, %1] : memref<2x128x128xf32>
      %7 = affine.load %arg3[%3, %2, %1] : memref<2x128x128xf32>
      %8 = addf %4, %5 : f32
      %9 = addf %8, %6 : f32
      %10 = addf %9, %7 : f32
      affine.store %10, %0[%3, %2, %1] : memref<2x128x128xf32>
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown6(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256x128xf32>, %arg3: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %1 = memref.alloc() : memref<2x128x128xf32>
    %2 = memref.collapse_shape %1 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %3 = memref.expand_shape %arg3 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %4 = memref.alloc() : memref<2x128x128xf32>
    %5 = memref.collapse_shape %4 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %6 = memref.expand_shape %arg2 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    affine.for %arg4 = 0 to 32768 {
      %7 = affine.apply #map0(%arg4)
      %8 = affine.apply #map2(%arg4)
      %9 = affine.apply #map3(%arg4)
      %10 = affine.load %0[%9, %8] : memref<2x128xi1>
      %11 = affine.load %arg1[%9, %8, %7] : memref<2x128x128xf32>
      %12 = affine.load %6[%9, %8, %7] : memref<2x128x128xf32>
      %13 = select %10, %11, %12 : f32
      affine.store %13, %4[%9, %8, %7] : memref<2x128x128xf32>
      %14 = affine.load %3[%9, %8] : memref<2x128xi1>
      %15 = select %14, %11, %12 : f32
      affine.store %15, %1[%9, %8, %7] : memref<2x128x128xf32>
    }
    return %5, %2 : memref<256x128xf32>, memref<256x128xf32>
  }
  func private @Unknown7(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) -> memref<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<128x128xf32>
    affine.for %arg3 = 0 to 16384 {
      %1 = affine.apply #map0(%arg3)
      %2 = affine.apply #map1(%arg3)
      %3 = affine.load %arg0[%2] : memref<128xi1>
      %4 = affine.load %arg1[%2, %1] : memref<128x128xf32>
      %5 = affine.load %arg2[%2, %1] : memref<128x128xf32>
      %6 = select %3, %4, %5 : f32
      affine.store %6, %0[%2, %1] : memref<128x128xf32>
    }
    return %0 : memref<128x128xf32>
  }
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<1x512xi64>, %arg2: memref<1x512xi64>, %arg3: memref<30522x128xf32>, %arg4: memref<2x128xf32>, %arg5: memref<512x128xf32>, %arg6: memref<128xf32>, %arg7: memref<128xf32>, %arg8: memref<128x128xf32>, %arg9: memref<128xf32>, %arg10: memref<128x128xf32>, %arg11: memref<128xf32>, %arg12: memref<128x128xf32>, %arg13: memref<128xf32>, %arg14: memref<2x1x1x128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<2x1x1x128xf32>, %arg32: memref<128x128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<512x128xf32>, %arg37: memref<512xf32>, %arg38: memref<128x512xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<30522xf32>, %arg47: memref<2x128x30522xf32>) -> (memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<128x128xf32>
    %1 = memref.alloc() : memref<512x128xf32>
    %2 = memref.alloc() : memref<128x128xf32>
    %3 = memref.alloc() : memref<2x128xf32>
    %4 = memref.alloc() : memref<30522x128xf32>
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
    %84 = memref.alloc() : memref<30522xf32>
    %85 = memref.alloc() : memref<30522x128xf32>
    %86 = memref.alloc() : memref<2x128x128xf32>
    %87 = memref.alloc() : memref<2x128x30522xf32>
    %88 = memref.alloc() : memref<256xf32>
    %89 = memref.alloc() : memref<256xf32>
    %90 = memref.alloc() : memref<2x128x128xf32>
    %91 = memref.alloc() : memref<0xf32>
    %92 = memref.alloc() : memref<2x128x128xf32>
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<2x128x128xf32>
    %95 = memref.alloc() : memref<256xf32>
    %96 = memref.alloc() : memref<256xf32>
    %97 = memref.alloc() : memref<2x128x128xf32>
    %98 = memref.alloc() : memref<2x128x128xui8>
    %99 = memref.alloc() : memref<2x128x128xf32>
    %100 = memref.alloc() : memref<2x128x128xf32>
    %101 = memref.alloc() : memref<0xf32>
    %102 = memref.alloc() : memref<2x128x512xf32>
    %103 = memref.alloc() : memref<2x128x512xf32>
    %104 = memref.alloc() : memref<2x128x128xf32>
    %105 = memref.alloc() : memref<256xf32>
    %106 = memref.alloc() : memref<256xf32>
    %107 = memref.alloc() : memref<2x128x128xf32>
    %108 = memref.alloc() : memref<2x128x128xui8>
    %109 = memref.alloc() : memref<2x128x128xf32>
    %110 = memref.alloc() : memref<2x128x128xf32>
    %111 = memref.alloc() : memref<2x128x128xf32>
    %112 = memref.alloc() : memref<2x128x2x64xf32>
    %113 = memref.alloc() : memref<2x2x128x64xf32>
    %114 = memref.alloc() : memref<2x2x128x64xf32>
    %115 = memref.alloc() : memref<2x2x128x128xui8>
    %116 = memref.alloc() : memref<2x2x128x128xf32>
    %117 = memref.alloc() : memref<2x2x128x128xf32>
    %118 = memref.alloc() : memref<2x2x128x128xf32>
    %119 = memref.alloc() : memref<2x2x128x64xf32>
    %120 = memref.alloc() : memref<2x2x128x64xf32>
    %121 = memref.alloc() : memref<2x128x128xf32>
    %122 = memref.alloc() : memref<256xf32>
    %123 = memref.alloc() : memref<256xf32>
    %124 = memref.alloc() : memref<2x128x128xf32>
    %125 = memref.alloc() : memref<2x128x128xui8>
    %126 = memref.alloc() : memref<2x128x128xf32>
    %127 = memref.alloc() : memref<2x128x128xf32>
    %128 = memref.alloc() : memref<0xf32>
    %129 = memref.alloc() : memref<2x128x512xf32>
    %130 = memref.alloc() : memref<2x128x512xf32>
    %131 = memref.alloc() : memref<2x128x128xf32>
    %132 = memref.alloc() : memref<256xf32>
    %133 = memref.alloc() : memref<256xf32>
    %134 = memref.alloc() : memref<2x128x128xf32>
    %135 = memref.alloc() : memref<2x128x128xui8>
    %136 = memref.alloc() : memref<2x128x128xf32>
    %137 = memref.alloc() : memref<2x128x128xf32>
    %138 = memref.alloc() : memref<2x128x128xf32>
    %139 = memref.alloc() : memref<2x128x2x64xf32>
    %140 = memref.alloc() : memref<2x2x128x64xf32>
    %141 = memref.alloc() : memref<2x2x128x64xf32>
    %142 = memref.alloc() : memref<2x2x128x128xui8>
    %143 = memref.alloc() : memref<2x2x128x128xf32>
    %144 = memref.alloc() : memref<2x2x128x128xf32>
    %145 = memref.alloc() : memref<2x2x128x128xf32>
    %146 = memref.alloc() : memref<2x2x128x64xf32>
    %147 = memref.alloc() : memref<2x2x128x64xf32>
    %148 = memref.alloc() : memref<2x128x128xf32>
    %149 = memref.alloc() : memref<256xf32>
    %150 = memref.alloc() : memref<256xf32>
    %151 = memref.alloc() : memref<2x128x128xf32>
    %152 = memref.alloc() : memref<1x128x128xf32>
    %153 = memref.alloc() : memref<128x128xf32>
    %154 = memref.alloc() : memref<256x128xf32>
    %155 = memref.alloc() : memref<256x128xf32>
    %156 = memref.alloc() : memref<1x128xi64>
    %157 = memref.alloc() : memref<128xi64>
    %158 = memref.alloc() : memref<1x128xi64>
    %159 = memref.alloc() : memref<30522x128xf32>
    %160 = memref.alloc() : memref<256xi64>
    %161 = memref.alloc() : memref<256xf64>
    %162 = memref.alloc() : memref<2x128xf32>
    %163 = memref.alloc() : memref<256xi64>
    %164 = memref.alloc() : memref<256xi64>
    %165 = memref.alloc() : memref<256xf64>
    %166 = memref.alloc() : memref<256x128xf32>
    %167 = memref.alloc() : memref<512x128xf32>
    %168 = memref.alloc() : memref<128xi64>
    %169 = memref.alloc() : memref<128xi64>
    %170 = memref.alloc() : memref<128xf64>
    %171 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128x128xf32>} : (memref<128x128xf32>) -> ()
    "lmhlo.constant"(%171) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%170) {value = dense<-1.000000e+00> : tensor<128xf64>} : (memref<128xf64>) -> ()
    "lmhlo.constant"(%169) {value = dense<512> : tensor<128xi64>} : (memref<128xi64>) -> ()
    "lmhlo.constant"(%168) {value = dense<0> : tensor<128xi64>} : (memref<128xi64>) -> ()
    "lmhlo.constant"(%167) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    "lmhlo.constant"(%166) {value = dense<0.000000e+00> : tensor<256x128xf32>} : (memref<256x128xf32>) -> ()
    "lmhlo.constant"(%165) {value = dense<-1.000000e+00> : tensor<256xf64>} : (memref<256xf64>) -> ()
    "lmhlo.constant"(%164) {value = dense<2> : tensor<256xi64>} : (memref<256xi64>) -> ()
    "lmhlo.constant"(%163) {value = dense<0> : tensor<256xi64>} : (memref<256xi64>) -> ()
    "lmhlo.constant"(%162) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    "lmhlo.constant"(%161) {value = dense<0.000000e+00> : tensor<256xf64>} : (memref<256xf64>) -> ()
    "lmhlo.constant"(%160) {value = dense<30522> : tensor<256xi64>} : (memref<256xi64>) -> ()
    "lmhlo.constant"(%159) {value = dense<0.000000e+00> : tensor<30522x128xf32>} : (memref<30522x128xf32>) -> ()
    "lmhlo.slice"(%arg1, %158) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.reshape"(%158, %157) : (memref<1x128xi64>, memref<128xi64>) -> ()
    "lmhlo.slice"(%arg2, %156) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %172:3 = call @Unknown0(%arg0, %163, %160, %161) : (memref<2x128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg3, %172#0, %155) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %173:3 = call @Unknown1(%157, %163, %164, %165) : (memref<128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg4, %173#0, %154) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %174 = call @Unknown2(%155, %154) : (memref<256x128xf32>, memref<256x128xf32>) -> memref<2x128x128xf32>
    %175:3 = call @Unknown3(%156, %168, %169, %170) : (memref<1x128xi64>, memref<128xi64>, memref<128xi64>, memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    "lmhlo.gather"(%arg5, %175#0, %153) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    "lmhlo.reshape"(%153, %152) : (memref<128x128xf32>, memref<1x128x128xf32>) -> ()
    "lmhlo.custom_call"(%174, %arg6, %arg7, %152, %151, %150, %149, %148) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%151, %arg8, %arg9, %147) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%151, %arg10, %arg11, %146) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%147, %146, %145) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%145, %arg14, %144, %143, %142) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%151, %arg12, %arg13, %141) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%143, %141, %140) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%140, %139) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%139, %138) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%138, %arg15, %arg16, %137, %136, %135) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%137, %arg17, %arg18, %151, %134, %133, %132, %131) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%134, %arg19, %arg20, %130, %129, %128) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%130, %arg21, %arg22, %127, %126, %125) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%127, %arg23, %arg24, %134, %124, %123, %122, %121) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%124, %arg25, %arg26, %120) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%124, %arg27, %arg28, %119) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%120, %119, %118) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%118, %arg31, %117, %116, %115) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%124, %arg29, %arg30, %114) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%116, %114, %113) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%113, %112) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%112, %111) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%111, %arg32, %arg33, %110, %109, %108) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%110, %arg34, %arg35, %124, %107, %106, %105, %104) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%107, %arg36, %arg37, %103, %102, %101) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%103, %arg38, %arg39, %100, %99, %98) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%100, %arg40, %arg41, %107, %97, %96, %95, %94) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%97, %arg42, %arg43, %93, %92, %91) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%93, %arg44, %arg45, %90, %89, %88) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.custom_call"(%90, %arg3, %arg46, %87) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<30522x128xf32>, memref<30522xf32>, memref<2x128x30522xf32>) -> ()
    "lmhlo.custom_call"(%arg47, %90, %arg3, %86, %85, %84) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x30522xf32>, memref<2x128x128xf32>, memref<30522x128xf32>, memref<2x128x128xf32>, memref<30522x128xf32>, memref<30522xf32>) -> ()
    "lmhlo.custom_call"(%86, %93, %arg44, %89, %88, %83, %82, %81) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%83, %97, %arg42, %92, %91, %80, %79, %78) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%80, %94, %arg40, %96, %95, %77, %76, %75, %74) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%77, %103, %arg38, %99, %98, %73, %72, %71) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%73, %107, %arg36, %102, %101, %70, %69, %68) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    "lmhlo.add"(%74, %70, %67) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%67, %104, %arg34, %106, %105, %66, %65, %64, %63) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%66, %111, %arg32, %109, %108, %62, %61, %60) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%62, %59) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%59, %58) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%58, %116, %114, %57, %56) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%57, %117, %115, %55) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%55, %120, %119, %54, %53) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%54, %124, %arg25, %52, %51, %50) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%56, %124, %arg29, %49, %48, %47) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%53, %124, %arg27, %46, %45, %44) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %176 = call @Unknown4(%63, %52, %49, %46) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%176, %121, %arg23, %123, %122, %43, %42, %41, %40) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%43, %130, %arg21, %126, %125, %39, %38, %37) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%39, %134, %arg19, %129, %128, %36, %35, %34) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    "lmhlo.add"(%40, %36, %33) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%33, %131, %arg17, %133, %132, %32, %31, %30, %29) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%32, %138, %arg15, %136, %135, %28, %27, %26) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%28, %25) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%25, %24) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%24, %143, %141, %23, %22) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%23, %144, %142, %21) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%21, %147, %146, %20, %19) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%20, %151, %arg8, %18, %17, %16) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%22, %151, %arg12, %15, %14, %13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%19, %151, %arg10, %12, %11, %10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %177 = call @Unknown5(%29, %18, %15, %12) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%177, %148, %arg6, %150, %149, %9, %8, %7, %6) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %178:2 = call @Unknown6(%172#2, %9, %166, %173#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    "lmhlo.scatter"(%159, %172#1, %178#0, %5) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %180 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%180) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    "lmhlo.add"(%85, %5, %4) : (memref<30522x128xf32>, memref<30522x128xf32>, memref<30522x128xf32>) -> ()
    "lmhlo.scatter"(%162, %173#1, %178#1, %3) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %180 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%180) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    "lmhlo.reduce"(%6, %171, %2) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %179 = call @Unknown7(%175#2, %2, %0) : (memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>) -> memref<128x128xf32>
    "lmhlo.scatter"(%167, %175#1, %179, %1) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %180 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%180) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    return %87, %4, %3, %1, %8, %7, %17, %16, %11, %10, %14, %13, %27, %26, %31, %30, %35, %34, %38, %37, %42, %41, %51, %50, %45, %44, %48, %47, %61, %60, %65, %64, %69, %68, %72, %71, %76, %75, %79, %78, %82, %81, %84 : memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}
