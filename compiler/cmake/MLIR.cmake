# SPDX-License-Identifier: Apache-2.0
# Modification Copyright (c) ByteDance Inc. All rights reserved.

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(BUILD_SHARED_LIBS ${LLVM_ENABLE_SHARED_LIBS} CACHE BOOL "" FORCE)
message(STATUS "BUILD_SHARED_LIBS       : " ${BUILD_SHARED_LIBS})

# ONNX uses exceptions, so LLVM_REQUIRES_EH might need ON, so that
# the functions from HandleLLVMOptions and AddLLVM don't disable exceptions.
# We currently disable exception and RTTI until we start supporting ONNX
#set(LLVM_REQUIRES_EH ON)
message(STATUS "LLVM_REQUIRES_EH        : " ${LLVM_REQUIRES_EH})

# If CMAKE_INSTALL_PREFIX was not provided explicitly and we are not using an install of
# LLVM and a CMakeCache.txt exists,
# force CMAKE_INSTALL_PREFIX to be the same as the LLVM build.
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND NOT LLVM_INSTALL_PREFIX)
  if (EXISTS ${LLVM_BINARY_DIR}/CMakeCache.txt)
    file(STRINGS ${LLVM_BINARY_DIR}/CMakeCache.txt prefix REGEX CMAKE_INSTALL_PREFIX)
    string(REGEX REPLACE "CMAKE_INSTALL_PREFIX:PATH=" "" prefix ${prefix})
    string(REGEX REPLACE "//.*" "" prefix ${prefix})
    set(CMAKE_INSTALL_PREFIX ${prefix} CACHE PATH "" FORCE)
  endif()
endif()
message(STATUS "CMAKE_INSTALL_PREFIX    : " ${CMAKE_INSTALL_PREFIX})

# Declare the library associated with a conversion.
function(add_byteir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY BYTEIR_CONVERSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_byteir_conversion_library)

# Declare the library associated with a translation.
function(add_byteir_translation_library name)
  set_property(GLOBAL APPEND PROPERTY BYTEIR_TRANSLATION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_byteir_translation_library)

# Declare the library associated with a statistics.
function(add_byteir_stat_library name)
  set_property(GLOBAL APPEND PROPERTY BYTEIR_STAT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_byteir_stat_library)