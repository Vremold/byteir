#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/torch-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/torch-frontend"
# path to torch-mlir
TORCH_MLIR_ROOT="$PROJ_DIR/third_party/torch-mlir"

function download_llvm_prebuilt() {
  pushd ${PROJ_DIR}
  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_ba8b8a73fcb6b830e63cd8e20c6e13b2a14d69bf.tar.gz"
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
  git submodule update -f $TORCH_MLIR_ROOT
  pushd $TORCH_MLIR_ROOT
  git apply ../patches/build.patch
  git apply ../patches/generated_torch_ops_td.patch
  git apply ../patches/one_hot.patch
  git apply ../patches/refine_types.patch
  git apply ../patches/torch_ops.patch
  popd
}

function prepare_for_build_with_prebuilt() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'

  pushd ${PROJ_DIR}
  # install requirements
  python3 -m pip install -r requirements.txt

  # init submodule
  git submodule update --init $TORCH_MLIR_ROOT
  pushd $TORCH_MLIR_ROOT
  git submodule update --init externals/mlir-hlo
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

  # init submodule
  git submodule update --init --recursive $TORCH_MLIR_ROOT

  # apply patches
  apply_patches

  unset http_proxy
  unset https_proxy
  unset no_proxy
}
