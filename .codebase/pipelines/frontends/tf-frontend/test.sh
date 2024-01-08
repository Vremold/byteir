#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/tf-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/tf-frontend"

function download_large_models() {
  pushd $ROOT_PROJ_DIR/..
  if [ ! -d bdaimodels ]; then
    git clone -b master --depth 1 git@code.byted.org:yuanhangjian/bdaimodelsv2.git bdaimodels
  fi
  cd bdaimodels
  git lfs pull --include tensorflow/recommender_model/industry_ecom_cvr_project_v50_spu_fm_bias_batch_r2042581_0.pb
  git lfs pull --include tensorflow/resnet/resnet50_v1.pb
  git lfs pull --include tensorflow/bert/bert-base-mrpc-without-preprocess.pb

  popd
  export TF_FRONTEND_BIN_PATH=$PROJ_DIR/bazel-bin/tools/tf-frontend
  export TF_LARGE_MODEL_PATH=$ROOT_PROJ_DIR/../bdaimodels/tensorflow
}

download_large_models

python3 -m pip install tensorflow==2.11.0 nvidia-cudnn-cu11==8.6.0.163
# export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib:/usr/local/cuda/lib64:
export TF_ENABLE_ONEDNN_OPTS=0
pushd $PROJ_DIR
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/compat PYTHONPATH=./build/python_packages/ python3 -m pytest $CUR_DIR/test_large_models.py
python3 -m pytest $CUR_DIR/test_large_models.py
popd
