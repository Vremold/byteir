#!/bin/bash

set -e

export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='mirrors.byted.org,apt.byted.org,bytedpypi.byted.org'

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/tf-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/tf-frontend"

git submodule update --init --recursive

BAZEL_VERSION=$(cat $ROOT_PROJ_DIR/external/tensorflow/.bazelversion)
wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-x86_64
mv bazel-$BAZEL_VERSION-linux-x86_64 /usr/bin/bazel -f
chmod +x /usr/bin/bazel
cp $PROJ_DIR/.tf_configure.bazelrc $ROOT_PROJ_DIR/external/tensorflow/.tf_configure.bazelrc

pushd $ROOT_PROJ_DIR/external/tensorflow
git apply ../patches/tensorflow/tf_build.patch
git apply ../patches/tensorflow/tf_dilated_conv.patch
git apply ../patches/tensorflow/tf_slice.patch
git apply ../patches/tensorflow/mhlo_ops.patch
git apply ../patches/tensorflow/grappler.patch
popd

pushd $PROJ_DIR
python3 -m pip install -r tf_mlir_ext/numerical/requirements.txt
bazel --output_user_root=./build build //tools:tf-frontend //tools:tf-ext-opt
bazel --output_user_root=./build test //tf_mlir_ext/tests:all //tf_mlir_ext/numerical:all --java_runtime_version=remotejdk_11
popd
