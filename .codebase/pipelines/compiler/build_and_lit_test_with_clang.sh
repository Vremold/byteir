#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."
# path to byteir/compiler
PROJ_DIR="$ROOT_PROJ_DIR/compiler"

# dir to build
BUILD_DIR="$PROJ_DIR/build_with_clang"
# dir to install
INSTALL_DIR="$BUILD_DIR/byre_install"

# build options
BYTEIR_ENABLE_ASAN=${BYTEIR_ENABLE_ASAN:-OFF}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

source $CUR_DIR/../prepare.sh
prepare_for_compiler_with_llvmraw

# install mhlo_tools
python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.2.2-cp39-cp39-linux_x86_64.whl

# build llvm
LLVM_DIR="$ROOT_PROJ_DIR/external/llvm-project"
LLVM_INSTALL_DIR="$LLVM_DIR/build/install"
pushd $LLVM_DIR
cmake -GNinja \
  "-H./llvm" \
  "-B./build" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_CCACHE_BUILD=OFF \
  -DMLIR_INCLUDE_TESTS=ON \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR"

cmake --build ./build --target all --target install
popd

# build byteir compiler
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cmake "-H$PROJ_DIR/cmake" \
      "-B$BUILD_DIR" \
      -GNinja \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
      -DLLVM_INSTALL_PATH="$LLVM_INSTALL_DIR" \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DBYTEIR_ENABLE_ASAN=${BYTEIR_ENABLE_ASAN} \
      -DBYTEIR_ENABLE_BINDINGS_PYTHON=ON
      # -DCMAKE_CXX_FLAGS="-Werror"

cmake --build "$BUILD_DIR" --target all check-byteir install
cmake --build "$BUILD_DIR" --target check-byteir-numerical

