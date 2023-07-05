#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $CUR_DIR/envsetup.sh

download_large_models
python3 -m pip install tensorflow==2.11.0 nvidia-cudnn-cu11==8.6.0.163
# export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib:/usr/local/cuda/lib64:
export TF_ENABLE_ONEDNN_OPTS=0
pushd $PROJ_DIR
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/compat PYTHONPATH=./build/python_packages/ python3 -m pytest $CUR_DIR/test_large_models.py
python3 -m pytest $CUR_DIR/test_large_models.py
popd
