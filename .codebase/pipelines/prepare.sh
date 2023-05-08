function download_llvm_prebuilt() {
  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_225d255a583ea3d50bbba49d949ca76be6a880bf.tar.gz"
    if [ ! -f "$LLVM_BUILD" ]; then
      rm -rf llvm_install*
      rm -rf llvm_build
      wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD" -q
      tar xzf "$LLVM_BUILD"
    fi
    LLVM_INSTALL_DIR="${PWD}/llvm_build"
  fi
}

function apply_patches() {
  pushd $ROOT_PROJ_DIR/external/mlir-hlo
  git clean -fd .
  for patch in $ROOT_PROJ_DIR/external/patches/mlir-hlo/*; do
    git apply $patch
  done
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
  git submodule update --init --recursive external/mlir-hlo external/AITemplate
  unset http_proxy; unset https_proxy; unset no_proxy

  # apply_patches
  download_llvm_prebuilt
  install_aitemplate
}

function prepare_for_runtime() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'
  git submodule update --init --recursive external/mlir-hlo external/cutlass external/date external/googletest external/pybind11
  unset http_proxy; unset https_proxy; unset no_proxy

  download_llvm_prebuilt
}
