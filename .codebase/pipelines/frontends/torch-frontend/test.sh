#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

source $CUR_DIR/envsetup.sh

pushd $PROJ_DIR
PYTHONPATH=./build/python_packages/ python3 -m pytest torch-frontend/python/test
popd
