#!/bin/bash

set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/tf-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/tf-frontend"

pushd $PROJ_DIR
TF_FRONTEND_COMMIT_ID="$(git rev-parse HEAD)"

rm -rf tf_frontend_download
mkdir tf_frontend_download
cp ./bazel-bin/tools/tf-frontend tf_frontend_download/
cp ./bazel-bin/tools/tf-ext-opt tf_frontend_download/
cp ./bazel-bin/external/org_tensorflow/tensorflow/libtensorflow_framework.so.2 tf_frontend_download/
tar -czf tf_frontend.tar.gz ./tf_frontend_download
wget http://tosv.byted.org/obj/tos-team/toscli/toscli -O toscli && chmod a+x toscli
./toscli put -name byteir/tf_frontend_${TF_FRONTEND_COMMIT_ID}.tar.gz ./tf_frontend.tar.gz

popd