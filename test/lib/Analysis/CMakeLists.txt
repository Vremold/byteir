# Exclude tests from libMLIR.so
add_mlir_library(ByteIRTestAnalysis
  TestPrintLiveness.cpp
  TestPrintSymbolicShape.cpp
  TestPrintUseRange.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ByteIRAnalysis
  LmhloDialect
  )