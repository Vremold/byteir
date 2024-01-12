#!/bin/bash

set -e
set -x

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."

pushd "$ROOT_PROJ_DIR/.codebase/pipelines/copybara"

# download prebuilt copybara.jar
wget http://tosv.byted.org/obj/turing/byteir/copybara_deploy.jar

# remove destination branch if it exists
if git ls-remote --exit-code --heads origin copybara_sync; then
  git push origin --delete copybara_sync
fi

# start sync
java -jar $(pwd)/copybara_deploy.jar $(pwd)/copy.bara.sky
popd