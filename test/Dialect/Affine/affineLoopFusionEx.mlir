// RUN: byteir-opt %s -affine-loop-fusion-ex | FileCheck %s

#map0 = affine_map<(d0) -> (d0 mod 128)>
#map1 = affine_map<(d0) -> ((d0 floordiv 128) mod 128)>
#map2 = affine_map<(d0) -> ((d0 floordiv 128) floordiv 128)>

// CHECK-LABEL: regular
func @regular(%arg0: memref<1x128xi64>, %arg1: memref<128xi64>, %arg2: memref<128xi64>, %arg3: memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) {
  %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
  %1 = memref.alloc() : memref<128xui32>
  affine.for %arg4 = 0 to 128 {
    %5 = affine.load %0[%arg4] : memref<128xi64>
    %6 = trunci %5 : i64 to i32
    %7 = builtin.unrealized_conversion_cast %6 : i32 to ui32
    affine.store %7, %1[%arg4] : memref<128xui32>
  }
  %2 = memref.alloc() : memref<128xi64>
  affine.for %arg4 = 0 to 128 {
    %5 = affine.load %0[%arg4] : memref<128xi64>
    %6 = affine.load %arg1[%arg4] : memref<128xi64>
    %7 = affine.load %arg2[%arg4] : memref<128xi64>
    %8 = addi %5, %7 : i64
    %9 = cmpi slt, %5, %6 : i64
    %10 = select %9, %8, %5 : i64
    affine.store %10, %2[%arg4] : memref<128xi64>
  }
  %3 = memref.expand_shape %2 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
  %4 = memref.alloc() : memref<128xi1>
  affine.for %arg4 = 0 to 128 {
    %5 = affine.load %0[%arg4] : memref<128xi64>
    %6 = affine.load %arg3[%arg4] : memref<128xf64>
    %7 = sitofp %5 : i64 to f64
    %8 = cmpf une, %7, %6 : f64
    affine.store %8, %4[%arg4] : memref<128xi1>
  }
  return %1, %3, %4 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
// CHECK: affine.for
// CHECK-NOT: affine.for
}

// CHECK-LABEL: withApply
func @withApply(%arg0: memref<128x128xf32>, %arg1: memref<128xf32>, %arg2: memref<1x128x128xf32>, %arg3: memref<1x128x128xf32>, %arg4: memref<1x128x128xf32>, %arg5: memref<1x128x128xf32>, %arg6: memref<1x128x128xf32>, %arg7: memref<1x128x128xf32>, %arg8: memref<1x128x128xf32>, %arg9: memref<1x128x128xf32>, %arg10: memref<1x128x128xf32>, %arg11: memref<1x128x128xf32>, %arg12: memref<1x128x128xf32>, %arg13: memref<1x128x128xf32>, %arg14: memref<1x128x128xf32>, %arg15: memref<1x128x128xf32>, %arg16: memref<1x128x128xf32>, %arg17: memref<1x128x128xf32>, %arg18: memref<1x128x128xf32>, %arg19: memref<1x128x128xf32>, %arg20: memref<1x128x128xf32>, %arg21: memref<1x128x128xf32>) -> (memref<1x128x128xf32>, memref<1x128x128xf32>) attributes {byre_elementwise_fusion} {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<128x128xf32> into memref<1x128x128xf32>
  %1 = memref.alloc() : memref<1x128x128xf32>
  affine.for %arg22 = 0 to 16384 {
    %3 = affine.apply #map0(%arg22)
    %4 = affine.apply #map1(%arg22)
    %5 = affine.apply #map2(%arg22)
    %6 = affine.load %0[%5, %4, %3] : memref<1x128x128xf32>
    %7 = affine.load %arg1[%3] : memref<128xf32>
    %8 = affine.load %arg2[%5, %4, %3] : memref<1x128x128xf32>
    %9 = affine.load %arg4[%5, %4, %3] : memref<1x128x128xf32>
    %10 = affine.load %arg3[%5, %4, %3] : memref<1x128x128xf32>
    %11 = affine.load %arg5[%5, %4, %3] : memref<1x128x128xf32>
    %12 = affine.load %arg6[%5, %4, %3] : memref<1x128x128xf32>
    %13 = affine.load %arg7[%5, %4, %3] : memref<1x128x128xf32>
    %14 = affine.load %arg8[%5, %4, %3] : memref<1x128x128xf32>
    %15 = affine.load %arg9[%5, %4, %3] : memref<1x128x128xf32>
    %16 = affine.load %arg10[%5, %4, %3] : memref<1x128x128xf32>
    %17 = affine.load %arg11[%5, %4, %3] : memref<1x128x128xf32>
    %18 = affine.load %arg12[%5, %4, %3] : memref<1x128x128xf32>
    %19 = affine.load %arg13[%5, %4, %3] : memref<1x128x128xf32>
    %20 = affine.load %arg14[%5, %4, %3] : memref<1x128x128xf32>
    %21 = affine.load %arg15[%5, %4, %3] : memref<1x128x128xf32>
    %22 = affine.load %arg16[%5, %4, %3] : memref<1x128x128xf32>
    %23 = affine.load %arg17[%5, %4, %3] : memref<1x128x128xf32>
    %24 = affine.load %arg18[%5, %4, %3] : memref<1x128x128xf32>
    %25 = affine.load %arg19[%5, %4, %3] : memref<1x128x128xf32>
    %26 = addf %6, %7 : f32
    %27 = mulf %26, %10 : f32
    %28 = minf %27, %11 : f32
    %29 = maxf %28, %9 : f32
    %30 = mulf %29, %29 : f32
    %31 = mulf %30, %12 : f32
    %32 = addf %31, %20 : f32
    %33 = mulf %32, %30 : f32
    %34 = addf %33, %21 : f32
    %35 = mulf %34, %30 : f32
    %36 = addf %35, %22 : f32
    %37 = mulf %36, %30 : f32
    %38 = addf %37, %23 : f32
    %39 = mulf %38, %30 : f32
    %40 = addf %39, %24 : f32
    %41 = addf %31, %13 : f32
    %42 = mulf %41, %30 : f32
    %43 = addf %42, %14 : f32
    %44 = mulf %43, %30 : f32
    %45 = addf %44, %15 : f32
    %46 = mulf %45, %30 : f32
    %47 = addf %46, %16 : f32
    %48 = mulf %47, %30 : f32
    %49 = addf %48, %17 : f32
    %50 = mulf %49, %30 : f32
    %51 = addf %50, %18 : f32
    %52 = mulf %51, %30 : f32
    %53 = addf %52, %19 : f32
    %54 = mulf %29, %53 : f32
    %55 = divf %54, %40 : f32
    %56 = addf %55, %25 : f32
    %57 = mulf %26, %8 : f32
    %58 = mulf %57, %56 : f32
    affine.store %58, %1[%5, %4, %3] : memref<1x128x128xf32>
  }
  %2 = memref.alloc() : memref<1x128x128xf32>
  affine.for %arg22 = 0 to 16384 {
    %3 = affine.apply #map0(%arg22)
    %4 = affine.apply #map1(%arg22)
    %5 = affine.apply #map2(%arg22)
    %6 = affine.load %arg4[%5, %4, %3] : memref<1x128x128xf32>
    %7 = affine.load %0[%5, %4, %3] : memref<1x128x128xf32>
    %8 = affine.load %arg1[%3] : memref<128xf32>
    %9 = affine.load %arg3[%5, %4, %3] : memref<1x128x128xf32>
    %10 = affine.load %arg5[%5, %4, %3] : memref<1x128x128xf32>
    %11 = affine.load %arg6[%5, %4, %3] : memref<1x128x128xf32>
    %12 = affine.load %arg7[%5, %4, %3] : memref<1x128x128xf32>
    %13 = affine.load %arg8[%5, %4, %3] : memref<1x128x128xf32>
    %14 = affine.load %arg9[%5, %4, %3] : memref<1x128x128xf32>
    %15 = affine.load %arg10[%5, %4, %3] : memref<1x128x128xf32>
    %16 = affine.load %arg11[%5, %4, %3] : memref<1x128x128xf32>
    %17 = affine.load %arg12[%5, %4, %3] : memref<1x128x128xf32>
    %18 = affine.load %arg13[%5, %4, %3] : memref<1x128x128xf32>
    %19 = affine.load %arg14[%5, %4, %3] : memref<1x128x128xf32>
    %20 = affine.load %arg15[%5, %4, %3] : memref<1x128x128xf32>
    %21 = affine.load %arg16[%5, %4, %3] : memref<1x128x128xf32>
    %22 = affine.load %arg17[%5, %4, %3] : memref<1x128x128xf32>
    %23 = affine.load %arg18[%5, %4, %3] : memref<1x128x128xf32>
    %24 = affine.load %arg19[%5, %4, %3] : memref<1x128x128xf32>
    %25 = affine.load %arg2[%5, %4, %3] : memref<1x128x128xf32>
    %26 = affine.load %arg20[%5, %4, %3] : memref<1x128x128xf32>
    %27 = affine.load %arg21[%5, %4, %3] : memref<1x128x128xf32>
    %28 = addf %7, %8 : f32
    %29 = mulf %28, %28 : f32
    %30 = mulf %29, %26 : f32
    %31 = math.exp %30 : f32
    %32 = mulf %28, %31 : f32
    %33 = mulf %32, %27 : f32
    %34 = mulf %28, %9 : f32
    %35 = minf %34, %10 : f32
    %36 = maxf %35, %6 : f32
    %37 = mulf %36, %36 : f32
    %38 = mulf %37, %11 : f32
    %39 = addf %38, %19 : f32
    %40 = mulf %39, %37 : f32
    %41 = addf %40, %20 : f32
    %42 = mulf %41, %37 : f32
    %43 = addf %42, %21 : f32
    %44 = mulf %43, %37 : f32
    %45 = addf %44, %22 : f32
    %46 = mulf %45, %37 : f32
    %47 = addf %46, %23 : f32
    %48 = addf %38, %12 : f32
    %49 = mulf %48, %37 : f32
    %50 = addf %49, %13 : f32
    %51 = mulf %50, %37 : f32
    %52 = addf %51, %14 : f32
    %53 = mulf %52, %37 : f32
    %54 = addf %53, %15 : f32
    %55 = mulf %54, %37 : f32
    %56 = addf %55, %16 : f32
    %57 = mulf %56, %37 : f32
    %58 = addf %57, %17 : f32
    %59 = mulf %58, %37 : f32
    %60 = addf %59, %18 : f32
    %61 = mulf %36, %60 : f32
    %62 = divf %61, %47 : f32
    %63 = addf %62, %24 : f32
    %64 = mulf %63, %25 : f32
    %65 = addf %64, %33 : f32
    affine.store %65, %2[%5, %4, %3] : memref<1x128x128xf32>
  }
  return %1, %2 : memref<1x128x128xf32>, memref<1x128x128xf32>
// CHECK: affine.for
// CHECK-NOT: affine.for
}