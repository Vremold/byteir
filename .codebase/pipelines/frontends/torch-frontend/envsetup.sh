#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/torch-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/torch-frontend"
# path to torch-mlir
TORCH_MLIR_ROOT="$PROJ_DIR/third_party/torch-mlir"

function download_llvm_prebuilt() {
  pushd ${PROJ_DIR}
  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_6875424135312aeb26ab8e0358ba7f9e6e80e741.tar.gz"
    if [ ! -f "$LLVM_BUILD" ]; then
      rm -rf llvm_install*
      rm -rf llvm_build
      wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD" -q
      tar xzf "$LLVM_BUILD"
    fi
    LLVM_INSTALL_DIR="${PWD}/llvm_build"
  fi
  popd
}

function apply_patches() {
  pushd $TORCH_MLIR_ROOT
  git clean -fd .
  for patch in ../patches/*; do
    git apply $patch
  done
  popd
}

function prepare_for_build_with_prebuilt() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'

  pushd ${PROJ_DIR}
  # install requirements
  python3 -m pip install -r requirements.txt
  python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.9-cp39-cp39-linux_x86_64.whl

  # init submodule
  git submodule update --init -f $TORCH_MLIR_ROOT
  pushd $TORCH_MLIR_ROOT
  git submodule update --init -f externals/mlir-hlo
  popd

  # apply patches
  apply_patches

  unset http_proxy
  unset https_proxy
  unset no_proxy
}

function prepare_for_build() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'

  pushd ${PROJ_DIR}
  # install requirements
  python3 -m pip install -r requirements.txt
  python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.9-cp39-cp39-linux_x86_64.whl

  # init submodule
  git submodule update --init --recursive -f $TORCH_MLIR_ROOT

  # apply patches
  apply_patches

  unset http_proxy
  unset https_proxy
  unset no_proxy
}

function download_large_models() {
  pushd $ROOT_PROJ_DIR/..
  if [ ! -d bdaimodels ]; then
    git clone -b master --depth 1 git@code.byted.org:yuanhangjian/bdaimodelsv2.git bdaimodels
  fi
  cd bdaimodels
  git lfs pull --include pytorch/sar_relevance_cross_model_latest/28365.ts
  git lfs pull --include pytorch/tt_label3_0607/torch_model_1654572315533.jit.revert.ts
  git lfs pull --include pytorch/swinv2_tiny/swinv2_tiny.pt
  git lfs pull --include pytorch/rtc1/torch_jit_1682337197499.jit.revert
  popd
  export TORCH_LARGE_MODEL_PATH=$ROOT_PROJ_DIR/../bdaimodels/pytorch
}
