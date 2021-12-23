// RUN: byteir-opt %s -func-tag="attach-attr=testAttr func-name=test" | FileCheck %s -check-prefix=DEFAULT
// RUN: byteir-opt %s -func-tag="attach-attr=testAttr:Unit func-name=test" | FileCheck %s -check-prefix=UNIT
// RUN: byteir-opt %s -func-tag="attach-attr=testAttr:String:test func-name=test" | FileCheck %s -check-prefix=STRING
// RUN: byteir-opt %s -func-tag="attach-attr=testAttr:I32:5 func-name=test" | FileCheck %s -check-prefix=I32
// RUN: byteir-opt %s -func-tag="attach-attr=testAttr:F32:2.5 func-name=test" | FileCheck %s -check-prefix=F32

func @test() {
    return
}
// DEFAULT: func @test() attributes {testAttr}
// UNIT: func @test() attributes {testAttr}
// STRING: func @test() attributes {testAttr = "test"}
// I32: func @test() attributes {testAttr = 5 : i32}
// F32: func @test() attributes {testAttr = 2.500000e+00 : f32}
