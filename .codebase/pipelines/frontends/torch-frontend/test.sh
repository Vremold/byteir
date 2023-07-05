#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $CUR_DIR/envsetup.sh


export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='*.byted.org'

pushd $PROJ_DIR
python3 -m pip install -r ./torch-requirements.txt
PYTHONPATH=./build/python_packages/ python3 -m pytest torch-frontend/python/test
popd

unset http_proxy
unset https_proxy
unset no_proxy

download_large_models
pushd $PROJ_DIR
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/compat PYTHONPATH=./build/python_packages/ python3 -m pytest $CUR_DIR/test_large_models.py
PYTHONPATH=./build/python_packages/ python3 -m pytest $CUR_DIR/test_large_models.py
popd
