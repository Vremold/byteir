#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."
# path to byteir/compiler
PROJ_DIR="$ROOT_PROJ_DIR/compiler"

# dir to build
BUILD_DIR="$PROJ_DIR/build"
# dir to install
INSTALL_DIR="$BUILD_DIR/byre_install"
# dir to ci artifact
OUT_DIR="$BUILD_DIR/artifact"

source $CUR_DIR/../common.sh
prepare_for_build

python3 -m pip install -r $PROJ_DIR/numerical/requirements.txt

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cmake "-H$PROJ_DIR/cmake" \
      "-B$BUILD_DIR" \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_PATH="$LLVM_INSTALL_DIR" \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

cmake --build "$BUILD_DIR" --config Release --target all check-byteir check-byteir-numerical install

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
tar -czvf "$OUT_DIR/byre_install.tar.gz" -C ${BUILD_DIR} byre_install
