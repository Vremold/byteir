// RUN: byteir-opt --canonicalize %s | FileCheck %s

func @test_ace_constant_case0() -> tensor<!ace.string> {
  %0 = "ace.constant"() {value = dense<"fork_active_pay"> : tensor<!ace.string>} : () -> tensor<!ace.string>
  return %0 : tensor<!ace.string>
}
// CHECK: ace.constant

func @test_ace_constant_case1() -> tensor<!ace.string> {
  %0 = ace.constant dense<"fork_active_pay"> : tensor<!ace.string>
  return %0 : tensor<!ace.string>
}
// CHECK: ace.constant