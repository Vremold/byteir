#!/bin/bash

set -e

export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='mirrors.byted.org,apt.byted.org,bytedpypi.byted.org'
git submodule update --init --recursive
unset http_proxy; unset https_proxy

LLVM_BUILD="llvm_install_579c4921c0105698617cc1e1b86a9ecf3b4717ce.tar.gz"
if [ ! -f "$LLVM_BUILD" ]; then
  rm -rf llvm_install*
  rm -rf llvm_build
  wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD"
  tar xzf "$LLVM_BUILD"
fi

mkdir build
cd build
cmake ../cmake/ -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_INSTALL_PATH=../llvm_build -DLLVM_EXTERNAL_LIT=$(which lit) \
        -DCMAKE_INSTALL_PREFIX=../byteir_build
cmake --build . --config Release --target all check-byteir install