// RUN: byteir-opt %s -remove-func-body="anchor-attr=testAttr" | FileCheck %s -check-prefix=DEFAULT
// RUN: byteir-opt %s -remove-func-body="anchor-attr=testAttr disable-force-private" | FileCheck %s -check-prefix=DISALBE


func private @test1() attributes {testAttr}  {
    return
}
// DEFAULT-LABEL: func private @test1()
// DEFAULT-NOT: return
// DISALBE-LABEL: func private @test1()
// DISALBE-NOT: return

func nested @test2() attributes {testAttr}  {
    return
}
// DEFAULT-LABEL: func nested @test2()
// DEFAULT-NOT: return
// DISALBE-LABEL: func nested @test2()
// DISALBE-NOT: return

func @test3() attributes {testAttr}  {
    return
}
// DEFAULT-LABEL: func private @test3()
// DEFAULT-NOT: return
// DISALBE-LABEL: func @test3()
// DISALBE-NEXT: return
