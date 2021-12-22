// RUN: byteir-opt %s -func-tag="attach-attr=testAttr func-name=test" | FileCheck %s

func @test() {
    return
}
// CHECK-LABEL: func @test() attributes {testAttr}

