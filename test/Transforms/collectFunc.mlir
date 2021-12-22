// RUN: byteir-opt %s -collect-func="anchor-attr=testAttr" | FileCheck %s


func private @test_private() {
    return
}
// CHECK-LABEL: func private @test_private() 

func @test1() attributes {testAttr} {
    return
}
// CHECK-LABEL: func @test1() attributes {testAttr}

func @test2() attributes {testAttr2} {
    return
}
// CHECK-NOT: func @test2() 

