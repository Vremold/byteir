set(MHLO_BUILD_EMBEDDED ON)
if (BYTEIR_ENABLE_BINDINGS_PYTHON)
  set(MLIR_ENABLE_BINDINGS_PYTHON ON)
  set(MHLO_ENABLE_BINDINGS_PYTHON ON)

  # FIXME(xrzhang):
  # Following code is about to hacking imported mlir targets on the group
  # of python sources to let cmake find correct path of those source files,
  # because of building mhlo python bindings were relied on the source of 
  # mlir python bindings and the searching rules of the source files on
  # imported targets are broken.
  # As mentioned above, this code snippet is highly dependent on the
  # impelementation details of mhlo python and mlir cmake build, and it
  # might be unmaintainable.
  if (LLVM_INSTALL_PATH)
    include(AddMLIRPython)
    function(relocate_python_source_root)
      cmake_parse_arguments(ARG "" "" "TARGETS" ${ARGN})
      function(_process_target target)
        set_target_properties(
          ${target} PROPERTIES
          mlir_python_ROOT_DIR "${LLVM_INSTALL_PATH}/src/python/${target}"
        )
      endfunction()
      _flatten_mlir_python_targets(_flat_targets ${ARG_TARGETS})
      foreach(sources_target ${_flat_targets})
        _process_target(${sources_target})
      endforeach()
    endfunction()

    relocate_python_source_root(
      TARGETS
      MLIRPythonSources
      MLIRPythonExtension.RegisterEverything)
  endif()
endif()

add_subdirectory(${REPO_ROOT_DIR}/external/mlir-hlo ${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo EXCLUDE_FROM_ALL)

include_directories(${REPO_ROOT_DIR}/external/mlir-hlo/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir-hlo/include)

install(DIRECTORY ${REPO_ROOT_DIR}/external/mlir-hlo/include/mlir-hlo ${REPO_ROOT_DIR}/external/mlir-hlo/include/mlir-hlo-c
  DESTINATION external/include
  COMPONENT byteir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  )

install(DIRECTORY ${CMAKE_BINARY_DIR}/mlir-hlo/include/mlir-hlo
  DESTINATION external/include
  COMPONENT byteir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )
