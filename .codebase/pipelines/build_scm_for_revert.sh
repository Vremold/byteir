#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../.."
OUTPUT_DIR="$ROOT_PROJ_DIR/output"

pushd $ROOT_PROJ_DIR/frontends/torch-frontend/torch-frontend/lib/CustomOp
rm -rf ./build
mkdir build

pushd build
cmake .. -DPython3_EXECUTABLE=python3
make -j4
cp ./libcustom_op.so $OUTPUT_DIR/
popd

popd

