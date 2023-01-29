#!/bin/bash

set -e

while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda)
      BRT_USE_CUDA=ON
      shift
      ;;
    --asan)
      BRT_ENABLE_ASAN=ON
      CMAKE_BUILD_TYPE=Debug
      shift
      ;;
    --python)
      BRT_ENABLE_PYTHON_BINDINGS=ON
      shift
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."
# path to byteir/runtime
PROJ_DIR="$ROOT_PROJ_DIR/runtime"

# dir to build
BUILD_DIR="$PROJ_DIR/build"
# dir to install
INSTALL_DIR="$BUILD_DIR/install"

if [[ -z ${BYRE_INSTALL_DIR} ]]; then
  BYRE_INSTALL_ARTIFACT=$ROOT_PROJ_DIR/artifact/byre_install.tar.gz
  if [[ ! -f ${BYRE_INSTALL_ARTIFACT} ]]; then
    echo "cannot find byre artifact"
    exit 1
  fi
  # dir to installed byre
  BYRE_INSTALL_DIR=${ROOT_PROJ_DIR}/byre_install
  rm -rf ${BYRE_INSTALL_DIR}
  tar xzf ${BYRE_INSTALL_ARTIFACT} -C ${ROOT_PROJ_DIR}
fi

# build options
BRT_USE_CUDA=${BRT_USE_CUDA:-OFF}
BRT_ENABLE_ASAN=${BRT_ENABLE_ASAN:-OFF}
BRT_ENABLE_PYTHON_BINDINGS=${BRT_ENABLE_PYTHON_BINDINGS:-OFF}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

source $CUR_DIR/../prepare.sh
prepare_for_runtime

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cmake -GNinja \
  "-H$PROJ_DIR/cmake" \
  "-B$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
  -DLLVM_INSTALL_PATH="$LLVM_INSTALL_DIR" \
  -DBYRE_INSTALL_PATH="$BYRE_INSTALL_DIR" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
  -Dbrt_USE_CUDA=${BRT_USE_CUDA} \
  -Dbrt_ENABLE_ASAN=${BRT_ENABLE_ASAN} \
  -Dbrt_ENABLE_PYTHON_BINDINGS=${BRT_ENABLE_PYTHON_BINDINGS}

cmake --build "$BUILD_DIR" --target all --target install

if [[ $BRT_USE_CUDA == "ON" ]] && [[ $BRT_ENABLE_ASAN == "ON" ]]; then
  export ASAN_OPTIONS=protect_shadow_gap=0
fi

pushd $BUILD_DIR
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/compat ./bin/brt_test_all
popd
