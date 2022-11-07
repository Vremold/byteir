// RUN: byteir-opt --host-opt -byre-opt %s | FileCheck %s
// CHECK-LABEL: func.func @main
module {
  func.func private @Unknown0(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi64>) -> memref<1xi32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1xi32>
    %1 = memref.load %arg2[%c0] : memref<1xi64>
    %2 = memref.load %arg0[%c0] : memref<1xi64>
    %3 = memref.load %arg1[%c0] : memref<1xi64>
    %4 = arith.addi %2, %3 : i64
    %5 = arith.addi %1, %4 : i64
    %6 = arith.trunci %5 : i64 to i32
    memref.store %6, %0[%c0] : memref<1xi32>
    return %0 : memref<1xi32>
  }
  func.func private @Unknown1(%arg0: memref<1xi32>, %arg1: memref<1x128xi32>) -> (memref<1x128xi32>, memref<1x128xi32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : tensor<1x128xi32>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c-1 = arith.constant -1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128xi32>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      %2 = arith.cmpi slt, %arg2, %c0 : index
      %3 = arith.addi %arg2, %c128 : index
      %4 = arith.select %2, %3, %arg2 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %2, %5, %arg2 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %2, %8, %7 : index
      %10 = tensor.extract %cst[%9, %4] : tensor<1x128xi32>
      %11 = memref.load %arg0[%9] : memref<1xi32>
      %12 = arith.cmpi slt, %10, %11 : i32
      %13 = arith.extui %12 : i1 to i32
      memref.store %13, %0[%9, %4] : memref<1x128xi32>
    }
    %1 = memref.alloc() {alignment = 128 : i64} : memref<1x128xi32>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      %2 = arith.cmpi slt, %arg2, %c0 : index
      %3 = arith.addi %arg2, %c128 : index
      %4 = arith.select %2, %3, %arg2 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %2, %5, %arg2 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %2, %8, %7 : index
      %10 = memref.load %0[%9, %4] : memref<1x128xi32>
      %11 = memref.load %arg1[%9, %4] : memref<1x128xi32>
      %12 = arith.muli %10, %11 : i32
      memref.store %12, %1[%9, %4] : memref<1x128xi32>
    }
    return %0, %1 : memref<1x128xi32>, memref<1x128xi32>
  }
  func.func @main(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi64>, %arg3: memref<1x128xi32>) -> (memref<1x128xi32>, memref<1x128xi32>) {
    %0 = call @Unknown0(%arg0, %arg1, %arg2) : (memref<1xi64>, memref<1xi64>, memref<1xi64>) -> memref<1xi32>
    %1:2 = call @Unknown1(%0, %arg3) : (memref<1xi32>, memref<1x128xi32>) -> (memref<1x128xi32>, memref<1x128xi32>)
    return %1#0, %1#1 : memref<1x128xi32>, memref<1x128xi32>
  }
}

