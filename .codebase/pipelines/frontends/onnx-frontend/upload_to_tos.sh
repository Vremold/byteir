#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/onnx-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/onnx-frontend"

function of_upload() {
  pushd $PROJ_DIR
  ONNX_FRONTEND_DOWNLOAD=$PROJ_DIR/onnx_frontend_download

  if [ -d ${ONNX_FRONTEND_DOWNLOAD} ]; then
    rm -rf ${ONNX_FRONTEND_DOWNLOAD}
  fi
  mkdir ${ONNX_FRONTEND_DOWNLOAD}

  cd ${PROJ_DIR}/build
  cmake --install . --prefix ${ONNX_FRONTEND_DOWNLOAD}

  cd ${PROJ_DIR}
  ONNX_FRONTEND_COMMIT_ID="$(git rev-parse HEAD)"

  tar -czvf onnx_frontend.tar.gz ${ONNX_FRONTEND_DOWNLOAD}
  wget http://tosv.byted.org/obj/tos-team/toscli/toscli -O toscli && chmod a+x toscli
  ./toscli put -name byteir/onnx_frontend_${ONNX_FRONTEND_COMMIT_ID}.tar.gz ./onnx_frontend.tar.gz

  rm toscli
  rm onnx_frontend.tar.gz
  popd
}

# Upload to TOS
of_upload
