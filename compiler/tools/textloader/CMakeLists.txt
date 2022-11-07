set(BYTEIR_LIBS
  MLIRByreDialect
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${dialect_libs}
  ${BYTEIR_LIBS}
  #MLIRLoopAnalysis
  #MLIRAnalysis
  MLIRDialect
 # MLIROptLib
  MLIRParser
  #MLIRPass
  #MLIRTransforms
  #MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

link_directories(${BYTEIR_BIN_LIB_DIR})
add_executable(textloader textloader.cpp)
add_dependencies(textloader ${LIBS})

llvm_update_compile_flags(textloader)
target_link_libraries(textloader PRIVATE ${LIBS})

mlir_check_all_link_libraries(textloader)
