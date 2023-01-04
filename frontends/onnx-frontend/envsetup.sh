#!/usr/bin/env bash

export BYTEIR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../.. &> /dev/null && pwd )"
export ONNX_FRONTEND_ROOT="$BYTEIR_ROOT/frontends/onnx-frontend"
echo "BYTEIR_ROOT = $BYTEIR_ROOT"
echo "ONNX_FRONTEND_ROOT = $ONNX_FRONTEND_ROOT"

function of_envsetup() {
  set -x
  set -e
  pushd $ONNX_FRONTEND_ROOT
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118'
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118'
  export no_proxy='mirrors.byted.org,apt.byted.org,bytedpypi.byted.org'

  apt-get update
  apt-get install protobuf-compiler -y
  apt-get install clang-format-13

  # install mhlo_tools
  MHLO_TOOLS_WHL="mhlo_tools-1.0.5-cp37-cp37m-linux_x86_64.whl"
  if [ -z "$( python3 -m pip freeze | grep $MHLO_TOOLS_WHL )" ]; then
    wget -q http://tosv.byted.org/obj/turing/byteir/${MHLO_TOOLS_WHL}
    python3 -m pip install ${MHLO_TOOLS_WHL}
    rm ${MHLO_TOOLS_WHL}
  fi

  LLVM_BUILD_HOME=$BYTEIR_ROOT/llvm_build_rtti

  # install requirements
  python3 -m pip install -r $ONNX_FRONTEND_ROOT/requirements.txt

  # download llvm-project
  if [ ! -d ${LLVM_BUILD_HOME} ]; then
    cd $(dirname ${LLVM_BUILD_HOME})
    LLVM_COMMIT_ID=74fb770de9399d7258a8eda974c93610cfde698e
    wget -q http://tosv.byted.org/obj/turing/byteir/llvm_install_rtti_${LLVM_COMMIT_ID}.tar.gz
    tar -xzf llvm_install_rtti_${LLVM_COMMIT_ID}.tar.gz
    rm llvm_install_rtti_${LLVM_COMMIT_ID}.tar.gz
  else
    echo "Directory ${LLVM_BUILD_HOME} already exists"
  fi

  # init submodule
  echo "Initializing Submodules"
  cd $ONNX_FRONTEND_ROOT
  git submodule update --init --recursive

  # # download onnx files for testing
  # cd $ONNX_FRONTEND_ROOT/..
  # apt-get install -y git-lfs
  # git lfs install --force --skip-smudge
  # if [ ! -d bdaimodels ]; then
  #   git clone git@code.byted.org:yuanhangjian/bdaimodelsv2.git bdaimodels
  # fi
  # cd bdaimodels
  # git lfs pull --include onnx/onnx_frontend/

  unset http_proxy; unset https_proxy
  popd
  set +e
  set +x
  echo "Done"
}

function of_build() {
  pushd $ONNX_FRONTEND_ROOT
  LLVM_BUILD_HOME=$BYTEIR_ROOT/llvm_build_rtti
  LLVM_LIT_PATH="$(which lit)"
  ONNX_MLIR_ROOT=$ONNX_FRONTEND_ROOT/third_party/onnx-mlir

  cd $ONNX_MLIR_ROOT
  if [[ $(git diff src/Transform/ONNX/ShapeInferencePass.cpp) ]]; then
      echo "Patch already applied to ShapeInferenPass.cpp"
  else
      echo "Applying a patch to ShapeInferecePass.cpp"
      git apply $ONNX_FRONTEND_ROOT/third_party/patches/ShapeInferencePass.patch
  fi

  if [ ! -d ${ONNX_FRONTEND_ROOT}/build ]; then
    mkdir ${ONNX_FRONTEND_ROOT}/build
  fi

  cd ${ONNX_FRONTEND_ROOT}/build

  cmake -G Ninja \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DPython3_ROOT_DIR=/usr/bin/python3.7 \
      -DPY_VERSION=3 \
      -DMLIR_DIR=${LLVM_BUILD_HOME}/lib/cmake/mlir \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_EXTERNAL_LIT=${LLVM_LIT_PATH} \
      ..

  cmake --build . --config Release
  popd
}

function of_test_lit() {
  pushd $ONNX_FRONTEND_ROOT
  ONNX_MLIR_ROOT=$ONNX_FRONTEND_ROOT/third_party/onnx-mlir

  cd $ONNX_MLIR_ROOT
  if [[ $(git diff src/Transform/ONNX/ShapeInferencePass.cpp) ]]; then
      echo "Patch already applied to ShapeInferenPass.cpp"
  else
      echo "Applying a patch to ShapeInferecePass.cpp"
      git apply ../patches/ShapeInferencePass.patch
  fi

  cd ${ONNX_FRONTEND_ROOT}/build

  cmake --build . --target check-of-lit
  popd
}

function of_test_models() {
  pushd $ONNX_FRONTEND_ROOT
  export TF_ENABLE_ONEDNN_OPTS=0

  python3 -m pytest $ONNX_FRONTEND_ROOT/test/ops/ -s

  if [ -d ${BYTEIR_ROOT}/../bdaimodels ]; then
    python3 -m pytest $ONNX_FRONTEND_ROOT/test/models/ -s
  fi

  unset TF_ENABLE_ONEDNN_OPTS
  popd
}

function of_format() {
  pushd $ONNX_FRONTEND_ROOT
  echo "Checking format of onnx-frontend..."
  find onnx-frontend/ -iname *.h -o -iname *.cpp | xargs clang-format-13 -i -style=file
  echo "onnx-frontend format check pass"
  popd
}

function of_upload() {
  pushd $ONNX_FRONTEND_ROOT
  ONNX_FRONTEND_DOWNLOAD=$BYTEIR_ROOT/onnx_frontend_download

  if [ -d ${ONNX_FRONTEND_DOWNLOAD} ]; then
    rm -rf ${ONNX_FRONTEND_DOWNLOAD}
  fi
  mkdir ${ONNX_FRONTEND_DOWNLOAD}

  cd ${ONNX_FRONTEND_ROOT}/build
  cmake --install . --prefix ${ONNX_FRONTEND_DOWNLOAD}

  cd ${ONNX_FRONTEND_ROOT}
  ONNX_FRONTEND_COMMIT_ID="$(git rev-parse HEAD)"

  tar -czvf onnx_frontend.tmp.tar.gz ${ONNX_FRONTEND_DOWNLOAD}
  wget http://tosv.byted.org/obj/tos-team/toscli/toscli -O toscli && chmod a+x toscli
  ./toscli put -name byteir/onnx_frontend_${ONNX_FRONTEND_COMMIT_ID}.tar.gz ./onnx_frontend.tmp.tar.gz

  rm toscli
  rm onnx_frontend.tmp.tar.gz
  popd
}
