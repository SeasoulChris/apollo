#!/usr/bin/env bash

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [ -z "$1" ]; then
  TARGET="//..."
else
  TARGET="$@"
fi

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

set -e

# TODO(xiaoxq): Retire after py_proto_library fixed in apollo.
function build_py_proto() {
  if [ -d "./py_proto" ];then
    rm -rf py_proto
  fi
  mkdir py_proto
  find modules/ cyber/ -name "*.proto" \
      | grep -v node_modules \
      | xargs /opt/apollo/sysroot/bin/protoc --python_out=py_proto
  find modules/ cyber/ -name "*_service.proto" \
      | grep -v node_modules \
      | xargs /usr/bin/python3 -m grpc_tools.protoc --proto_path=. --python_out=py_proto --grpc_python_out=py_proto
  find py_proto/* -type d -exec touch "{}/__init__.py" \;
}

# Build widely-used apollo modules.
pushd /apollo
  build_py_proto
  bazel build --distdir="/apollo/.cache/distdir" -c opt \
      //cyber/python/cyber_py3:record \
      $( bazel query 'kind("py_library", //...)' | grep pb2$ )
popd

if [ -f "WORKSPACE.bazel" ]; then
  echo "###### You are building with local pip-cache! ######"
fi

DISTDIR="/fuel/.cache/distdir"
mkdir -p "${DISTDIR}"
bazel build --distdir="${DISTDIR}" ${TARGET}
