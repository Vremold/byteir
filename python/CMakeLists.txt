add_custom_target(ByteIRPythonModule
    COMMAND "${CMAKE_COMMAND}" -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/byteir ${CMAKE_BINARY_DIR}/python/byteir)

add_custom_target(LinkMLIRHLOPythonModules
    COMMAND "${CMAKE_COMMAND}" -E create_symlink ${CMAKE_BINARY_DIR}/mlir-hlo/python_packages/mlir_hlo/mlir ${CMAKE_BINARY_DIR}/python/mlir
    DEPENDS MLIRHLOPythonModules)

add_dependencies(ByteIRPythonModule LinkMLIRHLOPythonModules)

add_subdirectory(test)
