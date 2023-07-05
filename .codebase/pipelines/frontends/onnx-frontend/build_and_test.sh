#!/bin/bash

set -x
set -e

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../../.."
# path to byteir/frontends/onnx-frontend
PROJ_DIR="$ROOT_PROJ_DIR/frontends/onnx-frontend"

source $CUR_DIR/envsetup.sh

US_DEV=false
while getopts ":d" opt; do
    case $opt in
        d)
            US_DEV=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

download_llvm_prebuilt_rtti $US_DEV
of_envsetup
of_build
of_test_lit
of_download_models
of_test_models