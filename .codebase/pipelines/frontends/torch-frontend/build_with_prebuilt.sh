#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

source $CUR_DIR/envsetup.sh

download_llvm_prebuilt
prepare_for_build_with_prebuilt

pushd $PROJ_DIR
cmake -S . \
      -B ./build \
      -GNinja \
      -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++

cmake --build ./build --target all
popd