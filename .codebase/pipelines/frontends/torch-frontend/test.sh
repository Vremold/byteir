#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='*.byted.org'

source $CUR_DIR/envsetup.sh

pushd $PROJ_DIR
PYTHONPATH=./build/python_packages/ python3 -m pytest torch-frontend/python/test
popd

unset http_proxy
unset https_proxy
unset no_proxy
