#!/bin/bash

set -e
set -x

export http_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export https_proxy='http://sys-proxy-rd-relay.byted.org:8118';
export no_proxy='*.byted.org';

function install_bazelisk() {
  if ping -w 2 -c 1 mono.byted.org &>/dev/null; then
    # If the Monorepo website is reachable, use the install script.
    curl -s "https://mono.byted.org/tools/install.sh" | bash -s -- "bazel/tool/bazelisk"
  else
    # Raw install from TTP
    OS_BRANCH="/$(uname -s | tr '[:upper:]' '[:lower:]')"
    if [ "${OS_BRANCH}" = '/linux' ]; then
      OS_BRANCH=''
    fi
    DOWNLOAD_URL="https://luban-source.tiktokd.org/repository/scm/api/v1/download_latest/?name=bazel/tool/bazelisk${OS_BRANCH}&arch=$(uname -m)"
    TMP_DIR="$(mktemp -d)"
    trap "rm -rf ${TMP_DIR}" EXIT
    curl -L --connect-timeout 10 \
      --max-time 60 \
      --retry 5 \
      --retry-delay 1 \
      --retry-max-time 10 \
      -H 'Cache-Control: no-cache, no-store' \
      -s "${DOWNLOAD_URL}" | tar -C "${TMP_DIR}" -zxf -
    bash "${TMP_DIR}/install.sh"
  fi
}

if [ ! -e /usr/local/bin/bazelisk ]; then
  install_bazelisk
fi

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# path to byteir root
ROOT_PROJ_DIR="$CUR_DIR/../../.."

pushd "$ROOT_PROJ_DIR/.codebase/pipelines/copybara"

echo yes | bazel run @bazelize//scripts/copybara:run -- https://github.com/bytedance/byteir.git $(pwd)/copy.bara.sky --force --init-history

popd