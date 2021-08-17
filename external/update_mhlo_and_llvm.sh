#!/bin/bash
set -e

if [[ $# -ne 1 ]] ; then
  echo "Usage: $0 <path/to/llvm-project>"
  exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd $DIR/mlir-hlo
git checkout master
git pull
popd

pushd $1
git fetch
git checkout $(cat $DIR/mlir-hlo/build_tools/llvm_version.txt)
popd