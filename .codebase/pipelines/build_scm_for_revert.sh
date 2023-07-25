#!/bin/bash

set -e
set -x

http_proxy=http://bj-rd-proxy.byted.org:3128 https_proxy=http://bj-rd-proxy.byted.org:3128 python3.7 -m pip install --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html torch==1.8.1+cpu


CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../.."
OUTPUT_DIR="$ROOT_PROJ_DIR/output"

pushd $ROOT_PROJ_DIR/frontends/torch-frontend/torch-frontend/lib/CustomOp
rm -rf ./build
mkdir build

export CUDACXX=/usr/local/cuda/bin/nvcc
pushd build
cmake .. -DPython3_EXECUTABLE=python3.7
make -j4

rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR
cp ./libcustom_op.so $OUTPUT_DIR/
popd

popd

