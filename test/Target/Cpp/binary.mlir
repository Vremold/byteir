// RUN: byteir-translate -emit-cpp %s | FileCheck %s

// CHECK-LABEL: binary_int
// CHECK-SAME: (int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]])
func @binary_int(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NEXT: int32_t [[V3:[^ ]*]] = [[V1]] + [[V2]];
  %0 = addi %arg0, %arg1: i32
  // CHECK-NEXT: int32_t [[V4:[^ ]*]] = [[V1]] - [[V3]];
  %1 = subi %arg0, %0: i32
  // CHECK-NEXT: int32_t [[V5:[^ ]*]] = [[V3]] * [[V4]];
  %2 = muli %0, %1: i32
  // CHECK-NEXT: int32_t [[V6:[^ ]*]] = [[V5]] / [[V4]];
  %3 = divi_signed %2, %1: i32
  // CHECK-NEXT: int32_t [[V7:[^ ]*]] = [[V6]] % [[V5]];
  %4 = remi_signed %3, %2: i32
  // CHECK-NEXT: int32_t [[V8:[^ ]*]] = [[V7]] << [[V6]];
  %5 = shift_left %4, %3: i32
  // CHECK-NEXT: int32_t [[V9:[^ ]*]] = [[V8]] >> [[V7]];
  %6 = shift_right_signed %5, %4: i32
  // CHECK-NEXT: int32_t [[V10:[^ ]*]] = [[V9]] & [[V8]];
  %7 = and %6, %5: i32
  // CHECK-NEXT: bool [[V11:[^ ]*]] = [[V1]] < [[V2]];
  %8 = cmpi "slt", %arg0, %arg1 : i32
  // CHECK-NEXT: bool [[V12:[^ ]*]] = [[V1]] == [[V2]];
  %9 = cmpi "eq", %arg0, %arg1 : i32
  // CHECK-NEXT: bool [[V12:[^ ]*]] = [[V1]] != [[V2]];
  %10 = cmpi "ne", %arg0, %arg1 : i32
  // CHECK-NEXT: bool [[V14:[^ ]*]] = [[V1]] > [[V2]];
  %11 = cmpi "sgt", %arg0, %arg1 : i32
  return %7 : i32
}

// CHECK-LABEL: binary_float
// CHECK-SAME: (float [[V1:[^ ]*]], float [[V2:[^ ]*]])
func @binary_float(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK-NEXT: float [[V3:[^ ]*]] = [[V1]] + [[V2]];
  %0 = addf %arg0, %arg1: f32
  // CHECK-NEXT: float [[V4:[^ ]*]] = [[V1]] - [[V3]];
  %1 = subf %arg0, %0: f32
  // CHECK-NEXT: float [[V5:[^ ]*]] = [[V3]] * [[V4]];
  %2 = mulf %0, %1: f32
  // CHECK-NEXT: float [[V6:[^ ]*]] = [[V5]] / [[V4]];
  %3 = divf %2, %1: f32
  // CHECK-NEXT: bool [[V7:[^ ]*]] = [[V1]] == [[V2]];
  %4 = cmpf "oeq", %arg0, %arg1 : f32
  // CHECK-NEXT: bool [[V8:[^ ]*]] = [[V1]] != [[V2]];
  %5 = cmpf "one", %arg0, %arg1 : f32
  // CHECK-NEXT: bool [[V9:[^ ]*]] = [[V1]] < [[V2]];
  %6 = cmpf "olt", %arg0, %arg1 : f32
  // CHECK-NEXT: bool [[V10:[^ ]*]] = [[V1]] > [[V2]];
  %7 = cmpf "ogt", %arg0, %arg1 : f32
  return %3 : f32
}

// CHECK-LABEL: binary_bool
// CHECK-SAME: (bool [[V1:[^ ]*]], bool [[V2:[^ ]*]])
func @binary_bool(%arg0 : i1, %arg1 : i1) -> i1 {
  // CHECK-NEXT: bool [[V3:[^ ]*]] = [[V1]] & [[V2]];
  %0 = and %arg0, %arg1: i1
  // CHECK-NEXT: bool [[V4:[^ ]*]] = [[V1]] | [[V3]];
  %1 = or %arg0, %0: i1
  // CHECK-NEXT: bool [[V5:[^ ]*]] = [[V3]] ^ [[V4]];
  %2 = xor  %0, %1: i1
  return %2 : i1
}
