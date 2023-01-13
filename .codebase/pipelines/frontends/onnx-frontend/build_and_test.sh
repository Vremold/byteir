#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/onnx-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/onnx-frontend"

source $CUR_DIR/envsetup.sh

download_llvm_prebuilt_rtti
of_envsetup
of_build
of_test_lit
of_test_models