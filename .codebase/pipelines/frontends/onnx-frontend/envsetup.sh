#!/bin/bash

export BYTEIR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../../../.. &> /dev/null && pwd )"
export ONNX_FRONTEND_ROOT="$BYTEIR_ROOT/frontends/onnx-frontend"
echo "BYTEIR_ROOT = $BYTEIR_ROOT"
echo "ONNX_FRONTEND_ROOT = $ONNX_FRONTEND_ROOT"

function download_llvm_prebuilt_rtti() {
  pushd $BYTEIR_ROOT
  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_rtti_9acc2f37bdfce08ca0c2faec03392db10d1bb7a9.tar.gz"
    if [ ! -f "$LLVM_BUILD" ]; then
      rm -rf llvm_install_rtti*
      rm -rf llvm_build_rtti
      wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD" -q
      tar xzf "$LLVM_BUILD"
    fi
    LLVM_INSTALL_DIR="${PWD}/llvm_build_rtti"
  fi
  popd
}

function of_envsetup() {
  pushd $ONNX_FRONTEND_ROOT
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118'
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118'
  export no_proxy='*.byted.org'

  # install requirements
  python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.7-cp37-cp37m-linux_x86_64.whl
  python3 -m pip install -r $ONNX_FRONTEND_ROOT/requirements.txt

  # init submodule
  ONNX_MLIR_ROOT=$ONNX_FRONTEND_ROOT/third_party/onnx-mlir
  git submodule update --init --recursive $ONNX_MLIR_ROOT
  git submodule update -f $ONNX_MLIR_ROOT
  pushd $ONNX_MLIR_ROOT
  git clean -fd .
  git apply ../patches/ConstantExpandReductionDequantize.patch
  git apply ../patches/Pad.patch
  git apply ../patches/ShapeInference.patch
  popd

  # download bdaimodels repo
  pushd $BYTEIR_ROOT/..
  apt-get update && apt-get install -y git-lfs
  git lfs install --force --skip-smudge
  if [ ! -d bdaimodels ]; then
    git clone git@code.byted.org:yuanhangjian/bdaimodelsv2.git bdaimodels
  fi
  cd bdaimodels
  git lfs pull --include onnx/onnx_frontend/
  popd

  unset http_proxy; unset https_proxy; unset no_proxy
  popd
}

function of_build() {
  LLVM_INSTALL_DIR=$BYTEIR_ROOT/llvm_build_rtti
  if [ ! -d ${ONNX_FRONTEND_ROOT}/build ]; then
    mkdir ${ONNX_FRONTEND_ROOT}/build
  fi

  cmake "-H$ONNX_FRONTEND_ROOT" \
      "-B$ONNX_FRONTEND_ROOT/build" \
      -GNinja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DPython3_ROOT_DIR=/usr/bin/python3.7 \
      -DPY_VERSION=3 \
      -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_EXTERNAL_LIT=$(which lit)

  cmake --build "$ONNX_FRONTEND_ROOT/build" --config Release --target onnx-frontend onnx-frontend-opt
}

function of_test_lit() {
  cmake --build "$ONNX_FRONTEND_ROOT/build" --target check-of-lit
}

function of_test_models() {
  pushd $ONNX_FRONTEND_ROOT
  export LARGE_MODEL_PATH=$BYTEIR_ROOT/../bdaimodels/onnx/onnx_frontend/
  python3 -m pytest $ONNX_FRONTEND_ROOT/test/ops -s
  python3 -m pytest $ONNX_FRONTEND_ROOT/test/models -s
  unset LARGE_MODEL_PATH
  popd
}

function of_format() {
  find $ONNX_FRONTEND_ROOT/onnx-frontend/ -iname *.h -o -iname *.cpp | xargs clang-format-13 -i -style=file
}
