#!/bin/bash

set -e
set -x

US_DEV=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --asan)
            BYTEIR_ENABLE_ASAN=ON
            CMAKE_BUILD_TYPE=Debug
            shift
            ;;
        -d)
            US_DEV=true
            shift
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."
# path to byteir/compiler
PROJ_DIR="$ROOT_PROJ_DIR/compiler"

# dir to build
BUILD_DIR="$PROJ_DIR/build"
# dir to install
INSTALL_DIR="$BUILD_DIR/byre_install"

# build options
BYTEIR_ENABLE_ASAN=${BYTEIR_ENABLE_ASAN:-OFF}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

source $CUR_DIR/../prepare.sh
prepare_for_compiler $US_DEV

if [[ $US_DEV = false ]]; then
    python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.11-cp39-cp39-linux_x86_64.whl
else
    http_proxy='http://sys-proxy-rd-relay.byted.org:8118' https_proxy='http://sys-proxy-rd-relay.byted.org:8118' python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.11-cp39-cp39-linux_x86_64.whl
fi


rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cmake "-H$PROJ_DIR/cmake" \
      "-B$BUILD_DIR" \
      -GNinja \
      -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
      -DLLVM_INSTALL_PATH="$LLVM_INSTALL_DIR" \
      -DLLVM_EXTERNAL_LIT=$(which lit) \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DBYTEIR_ENABLE_ASAN=${BYTEIR_ENABLE_ASAN} \
      -DBYTEIR_ENABLE_BINDINGS_PYTHON=ON

cmake --build "$BUILD_DIR" --target all check-byteir install
cmake --build "$BUILD_DIR" --target check-byteir-numerical
cmake --build "$BUILD_DIR" --target byteir-python-pack

# TODO: make this test more robust
# test byteir.compile
pushd $ROOT_PROJ_DIR
PYTHONPATH=./compiler/build/python_packages/byteir python3 -m byteir.tools.compiler ./compiler/test/E2E/MLPInference/input.mlir ./test.mlir --entry_func forward
rm -f ./test.mlir
rm -f ./test.mlir.ptx
popd
