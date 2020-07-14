#!/usr/bin/env bash

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

cd /apollo
build_py_proto
bazel build -c opt \
    //cyber/python/cyber_py3:record \
    //modules/drivers/video/tools/decode_video/... \
    //modules/localization/msf/local_tool/data_extraction/... \
    //modules/localization/msf/local_tool/map_creation/... \
    //modules/map/tools:sim_map_generator \
    //modules/planning/pipeline/... \
    //modules/prediction/pipeline/... \
    //modules/routing/topo_creator \
    $( bazel query 'kind("py_library", //...)' | grep pb2$ )

SIMULATOR="/apollo-simulator"
if [ -d "${SIMULATOR}" ]; then
  pushd "${SIMULATOR}"
  bash build.sh build_tool
  bazel build -c opt \
      //sim_control/dynamic_model/echo_lincoln_pipeline:echo_lincoln_bin
  popd
fi
