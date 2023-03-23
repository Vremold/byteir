#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/tf-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/tf-frontend"

export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='*.byted.org'

bash $PROJ_DIR/scripts/prepare.sh

# build and test
pushd $PROJ_DIR
python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.8-cp39-cp39-linux_x86_64.whl
$PROJ_DIR/bazel --output_user_root=./build build //tools:tf-frontend //tools:tf-ext-opt
$PROJ_DIR/bazel --output_user_root=./build test //tf_mlir_ext/tests:all --java_runtime_version=remotejdk_11
$PROJ_DIR/bazel --output_user_root=./build test //tf_mlir_ext/numerical:all --java_runtime_version=remotejdk_11
popd

unset http_proxy; unset https_proxy; unset no_proxy