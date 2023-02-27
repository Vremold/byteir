#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

source $CUR_DIR/envsetup.sh

pushd $PROJ_DIR/build/torch-frontend/python/dist
wget http://tosv.byted.org/obj/tos-team/toscli/toscli -O toscli && chmod a+x toscli
for i in *.whl; do
    [ -f "$i" ] || break
    ./toscli put -name byteir/$i ./$i
done
popd
