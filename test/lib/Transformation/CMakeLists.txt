# Exclude tests from libMLIR.so
add_mlir_library(ByteIRTestTransformation
  TestConvertInsertion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ByteIRMhloPasses
  ByteIRUtils
  MhloDialect
  )