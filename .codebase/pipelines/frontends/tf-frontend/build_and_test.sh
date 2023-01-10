#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/tf-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/tf-frontend"

export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='*.byted.org'

git submodule update --init --recursive $PROJ_DIR/external/tensorflow
git submodule update -f $PROJ_DIR/external/tensorflow

# configure bazel
BAZEL_VERSION=$(cat $PROJ_DIR/external/tensorflow/.bazelversion)
if [ ! -f "bazel-$BAZEL_VERSION-linux-x86_64" ]; then
  wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-x86_64
fi
cp bazel-$BAZEL_VERSION-linux-x86_64 $PROJ_DIR/bazel -f
chmod +x $PROJ_DIR/bazel
cp $PROJ_DIR/.tf_configure.bazelrc $PROJ_DIR/external/tensorflow/.tf_configure.bazelrc

# apply patches
pushd $PROJ_DIR/external/tensorflow
git apply ../patches/tensorflow/tf_build.patch
git apply ../patches/tensorflow/tf_dilated_conv.patch
git apply ../patches/tensorflow/tf_slice.patch
git apply ../patches/tensorflow/mhlo_ops.patch
git apply ../patches/tensorflow/grappler.patch
popd

# build and test
pushd $PROJ_DIR
python3 -m pip install https://tosv.byted.org/obj/turing/byteir/mhlo_tools-1.0.6-cp37-cp37m-linux_x86_64.whl
$PROJ_DIR/bazel --output_user_root=./build build //tools:tf-frontend //tools:tf-ext-opt
$PROJ_DIR/bazel --output_user_root=./build test //tf_mlir_ext/tests:all --java_runtime_version=remotejdk_11
$PROJ_DIR/bazel --output_user_root=./build test //tf_mlir_ext/numerical:all --java_runtime_version=remotejdk_11
popd

unset http_proxy; unset https_proxy; unset no_proxy