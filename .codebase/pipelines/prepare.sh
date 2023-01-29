function download_llvm_prebuilt() {
  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_7ccbb4dff10efe6c26219204e361ddb0264938b8.tar.gz"
    if [ ! -f "$LLVM_BUILD" ]; then
      rm -rf llvm_install*
      rm -rf llvm_build
      wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD" -q
      tar xzf "$LLVM_BUILD"
    fi
    LLVM_INSTALL_DIR="${PWD}/llvm_build"
  fi
}

function prepare_for_compiler() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'
  git submodule update --init --recursive external/mlir-hlo
  unset http_proxy; unset https_proxy; unset no_proxy

  download_llvm_prebuilt
}

function prepare_for_runtime() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='*.byted.org'
  git submodule update --init --recursive external/mlir-hlo external/cutlass external/date external/googletest external/pybind11
  unset http_proxy; unset https_proxy; unset no_proxy

  download_llvm_prebuilt
}
