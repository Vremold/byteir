#!/bin/bash

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