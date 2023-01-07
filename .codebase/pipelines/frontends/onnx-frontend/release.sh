#!/bin/bash

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."

source $ROOT_PROJ_DIR/frontends/onnx-frontend/envsetup.sh
# For "git am"
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
# To stop pip version warning
python3 -m pip install --upgrade pip

of_envsetup

set -x
set -e
# Build onnx-frontend and test it
of_build
of_test_lit
# Upload to TOS
of_upload
