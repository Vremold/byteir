#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/tf-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/tf-frontend"

source $CUR_DIR/../../prepare.sh
prepare_for_tf_frontend

git submodule update -f $ROOT_PROJ_DIR/external/tensorflow

# configure bazel
BAZEL_VERSION=$(cat $ROOT_PROJ_DIR/external/tensorflow/.bazelversion)
if [ ! -f "bazel-$BAZEL_VERSION-linux-x86_64" ]; then
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-x86_64
  unset http_proxy; unset https_proxy
fi
mv bazel-$BAZEL_VERSION-linux-x86_64 $PROJ_DIR/bazel -f
chmod +x $PROJ_DIR/bazel
cp $PROJ_DIR/.tf_configure.bazelrc $ROOT_PROJ_DIR/external/tensorflow/.tf_configure.bazelrc

# apply patches
pushd $ROOT_PROJ_DIR/external/tensorflow
git apply ../patches/tensorflow/tf_build.patch
git apply ../patches/tensorflow/tf_dilated_conv.patch
git apply ../patches/tensorflow/tf_slice.patch
git apply ../patches/tensorflow/mhlo_ops.patch
git apply ../patches/tensorflow/grappler.patch
popd

# build and test
export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='mirrors.byted.org,apt.byted.org,bytedpypi.byted.org'
pushd $PROJ_DIR
python3 -m pip install -r tf_mlir_ext/numerical/requirements.txt
$PROJ_DIR/bazel --output_user_root=./build build //tools:tf-frontend //tools:tf-ext-opt
$PROJ_DIR/bazel --output_user_root=./build test //tf_mlir_ext/tests:all //tf_mlir_ext/numerical:all --java_runtime_version=remotejdk_11
popd
unset http_proxy; unset https_proxy