#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."

pushd $ROOT_PROJ_DIR
export CUDACXX=/usr/local/cuda/bin/nvcc
# build compiler
bash .codebase/pipelines/compiler/build_and_lit_test.sh
# build runtime
bash .codebase/pipelines/runtime/build_and_test.sh --cuda --python --no-test
# build torch_frontend
bash .codebase/pipelines/frontends/torch-frontend/build_with_prebuilt.sh

pip3 install $ROOT_PROJ_DIR/external/AITemplate/python/dist/*.whl 
pip3 install $ROOT_PROJ_DIR/compiler/build/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/runtime/python/dist/*.whl
pip3 install $ROOT_PROJ_DIR/frontends/torch-frontend/build/torch-frontend/python/dist/*.whl 
http_proxy='http://sys-proxy-rd-relay.byted.org:8118' https_proxy='http://sys-proxy-rd-relay.byted.org:8118' pip3 install -r $ROOT_PROJ_DIR/frontends/torch-frontend/torch-requirements.txt

python3 tests/numerical_test/main.py
rm -rf ./local_test
popd
