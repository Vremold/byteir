#!/bin/bash

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."

source $ROOT_PROJ_DIR/frontends/onnx-frontend/envsetup.sh

of_envsetup

set -x
set -e
# Build onnx-frontend and test it
of_build
of_test_lit
of_test_models
