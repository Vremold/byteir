function prepare_for_build() {
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
  export no_proxy='mirrors.byted.org,apt.byted.org,bytedpypi.byted.org'
  git submodule update --init --recursive
  unset http_proxy; unset https_proxy

  if [[ -z ${LLVM_INSTALL_DIR} ]]; then
    LLVM_BUILD="llvm_install_74fb770de9399d7258a8eda974c93610cfde698e.tar.gz"
    if [ ! -f "$LLVM_BUILD" ]; then
      rm -rf llvm_install*
      rm -rf llvm_build
      wget "http://tosv.byted.org/obj/turing/byteir/$LLVM_BUILD"
      tar xzf "$LLVM_BUILD"
    fi
    LLVM_INSTALL_DIR="${PWD}/llvm_build"
  fi
}