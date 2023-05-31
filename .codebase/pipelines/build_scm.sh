#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../.."
OUTPUT_DIR="$ROOT_PROJ_DIR/output"

pushd $ROOT_PROJ_DIR

export CUDACXX=/usr/local/cuda/bin/nvcc
# build byteir
bash $ROOT_PROJ_DIR/.codebase/pipelines/compiler/build_and_lit_test.sh
# build brt
bash $ROOT_PROJ_DIR/.codebase/pipelines/runtime/build_and_test.sh --cuda --python --no-test
# build torch-frontend
CMAKE_BUILD_PARALLEL_LEVEL=4 bash $ROOT_PROJ_DIR/.codebase/pipelines/frontends/torch-frontend/build_with_prebuilt.sh

mkdir $OUTPUT_DIR
cp $ROOT_PROJ_DIR/external/AITemplate/python/dist/*.whl $OUTPUT_DIR/
cp $ROOT_PROJ_DIR/compiler/build/python/dist/*.whl $OUTPUT_DIR/
cp $ROOT_PROJ_DIR/runtime/build/python/_brt.so $OUTPUT_DIR/
cp $ROOT_PROJ_DIR/runtime/build/lib/libbrt.so $OUTPUT_DIR/
cp $ROOT_PROJ_DIR/frontends/torch-frontend/build/torch-frontend/python/dist/*.whl $OUTPUT_DIR/

popd
