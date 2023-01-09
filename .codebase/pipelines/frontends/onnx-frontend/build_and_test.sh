#!/bin/bash

function download_llvm_prebuilt() {
  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_rtti_74fb770de9399d7258a8eda974c93610cfde698e.tar.gz"
    if [ ! -f "$LLVM_BUILD" ]; then
      rm -rf llvm_install_rtti*
      rm -rf llvm_build_rtti
      wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD"
      tar xzf "$LLVM_BUILD"
    fi
    LLVM_INSTALL_DIR="${PWD}/llvm_build_rtti"
  fi
}

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/onnx-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/onnx-frontend"
export ONNX_FRONTEND_ROOT="$PROJ_DIR"

export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='*.byted.org'

# install python dependency
python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.6-cp37-cp37m-linux_x86_64.whl
python3 -m pip install -r $PROJ_DIR/requirements.txt

git submodule update --init --recursive $PROJ_DIR/third_party/onnx-mlir
pushd $PROJ_DIR/third_party/onnx-mlir
git apply ../patches/ClipAvgpoolConvtransposeElementwise.patch
git apply ../patches/Pad.patch
git apply ../patches/ShapeInference.patch
popd

download_llvm_prebuilt

# Build onnx-frontend and test it
rm -rf "$PROJ_DIR/build"
mkdir -p "$PROJ_DIR/build"
cmake "-H$PROJ_DIR" \
      "-B$PROJ_DIR/build" \
      -GNinja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DPython3_ROOT_DIR=/usr/bin/python3.7 \
      -DPY_VERSION=3 \
      -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_EXTERNAL_LIT=$(which lit)

cmake --build "$PROJ_DIR/build" --config Release --target onnx-frontend onnx-frontend-opt check-of-lit

function of_test_models() {
  pushd $PROJ_DIR
  export TF_ENABLE_ONEDNN_OPTS=0
  python3 -m pytest "$PROJ_DIR/test/" -s
  popd
}

of_test_models

unset http_proxy; unset https_proxy; unset no_proxy