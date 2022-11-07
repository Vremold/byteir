// RUN: byteir-opt --to-llvm %s | FileCheck %s

// CHECK-LABEL: Unknown1
// CHECK-LABEL: Unknown0
module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    func.func @Unknown1(%arg0: memref<1xi32>, %arg1: memref<1x128xi32>, %arg2: memref<1x128xi32>, %arg3: memref<1x128xi32>) attributes {llvm.emit_c_interface} {
      %cst = arith.constant dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : tensor<1x128xi32>
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      %c-1 = arith.constant -1 : index
      scf.for %arg4 = %c0 to %c128 step %c1 {
        %0 = arith.cmpi slt, %arg4, %c0 : index
        %1 = arith.addi %arg4, %c128 : index
        %2 = arith.select %0, %1, %arg4 : index
        %3 = arith.subi %c-1, %arg4 : index
        %4 = arith.select %0, %3, %arg4 : index
        %5 = arith.divsi %4, %c128 : index
        %6 = arith.subi %c-1, %5 : index
        %7 = arith.select %0, %6, %5 : index
        %8 = tensor.extract %cst[%7, %2] : tensor<1x128xi32>
        %9 = memref.load %arg0[%7] : memref<1xi32>
        %10 = arith.cmpi slt, %8, %9 : i32
        %11 = arith.extui %10 : i1 to i32
        memref.store %11, %arg2[%7, %2] : memref<1x128xi32>
      }
      scf.for %arg4 = %c0 to %c128 step %c1 {
        %0 = arith.cmpi slt, %arg4, %c0 : index
        %1 = arith.addi %arg4, %c128 : index
        %2 = arith.select %0, %1, %arg4 : index
        %3 = arith.subi %c-1, %arg4 : index
        %4 = arith.select %0, %3, %arg4 : index
        %5 = arith.divsi %4, %c128 : index
        %6 = arith.subi %c-1, %5 : index
        %7 = arith.select %0, %6, %5 : index
        %8 = memref.load %arg2[%7, %2] : memref<1x128xi32>
        %9 = memref.load %arg1[%7, %2] : memref<1x128xi32>
        %10 = arith.muli %8, %9 : i32
        memref.store %10, %arg3[%7, %2] : memref<1x128xi32>
      }
      return
    }
    func.func @Unknown0(%arg0: memref<1xi64>, %arg1: memref<1xi64>, %arg2: memref<1xi64>, %arg3: memref<1xi32>) attributes {llvm.emit_c_interface} {
      %c0 = arith.constant 0 : index
      %0 = memref.load %arg2[%c0] : memref<1xi64>
      %1 = memref.load %arg0[%c0] : memref<1xi64>
      %2 = memref.load %arg1[%c0] : memref<1xi64>
      %3 = arith.addi %1, %2 : i64
      %4 = arith.addi %0, %3 : i64
      %5 = arith.trunci %4 : i64 to i32
      memref.store %5, %arg3[%c0] : memref<1xi32>
      return
    }
  }
  func.func @main(%arg0: memref<1xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x128xi32> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<1x128xi32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg5: memref<1x128xi32> {byre.argname = "Output1", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<4xi8>
    %1 = "byre.alias"(%0) {offset = 0 : i64} : (memref<4xi8>) -> memref<1xi32>
    byre.compute @LLVMJITOp(%arg0, %arg1, %arg2, %1) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1xi64>, memref<1xi64>, memref<1xi64>, memref<1xi32>
    byre.compute @LLVMJITOp(%1, %arg3, %arg4, %arg5) {kernel_name = "Unknown1", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<1xi32>, memref<1x128xi32>, memref<1x128xi32>, memref<1x128xi32>
    return
  }
}

