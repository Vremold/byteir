function download_llvm_prebuilt() {
  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_4592543a01609feb4b3c19e81a9d54743e15e329.tar.gz"
    if [ ! -f "$LLVM_BUILD" ]; then
      rm -rf llvm_install*
      rm -rf llvm_build
      if [[ $1 == false ]]; then
        wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD" -q
      else
        http_proxy='http://sys-proxy-rd-relay.byted.org:8118' https_proxy='http://sys-proxy-rd-relay.byted.org:8118' wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD" -q
      fi
      tar xzf "$LLVM_BUILD"
    fi
    LLVM_INSTALL_DIR="${PWD}/llvm_build"
  fi
}

function apply_mhlo_patches() {
  pushd $ROOT_PROJ_DIR/external/mlir-hlo
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/mlir-hlo/*; do
    git apply $patch
  done
  popd
}

function apply_aitemplate_patches() {
  pushd $ROOT_PROJ_DIR/external/AITemplate
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/AITemplate/*; do
    git apply $patch
  done
  popd
}

function install_aitemplate() {
  pushd external/AITemplate/python
  python3 setup.py bdist_wheel
  python3 -m pip uninstall -y aitemplate
  python3 -m pip install dist/*.whl
  popd
}

function prepare_for_compiler() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'
  git submodule update --init --recursive -f external/mlir-hlo external/AITemplate
  unset http_proxy; unset https_proxy; unset no_proxy

  apply_aitemplate_patches
  download_llvm_prebuilt $1
  install_aitemplate
}

function prepare_for_compiler_with_llvmraw() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'
  git submodule update --init --recursive -f external/mlir-hlo external/llvm-project
  unset http_proxy; unset https_proxy; unset no_proxy
}

function prepare_for_runtime() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'
  git submodule update --init --recursive -f external/mlir-hlo external/cutlass external/date external/googletest external/pybind11
  unset http_proxy; unset https_proxy; unset no_proxy

  download_llvm_prebuilt $1
}
