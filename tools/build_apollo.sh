#!/usr/bin/env bash

cd /apollo
bash apollo.sh build_py
bazel build -c opt \
    //cyber/py_wrapper:_cyber_record_py3.so \
    //modules/drivers/video/tools/decode_video/... \
    //modules/localization/msf/local_tool/data_extraction/... \
    //modules/localization/msf/local_tool/map_creation/... \
    //modules/map/tools:sim_map_generator \
    //modules/planning/pipeline/... \
    //modules/prediction/pipeline/... \
    //modules/routing/topo_creator

SIMULATOR="/apollo-simulator"
if [ -d "${SIMULATOR}" ]; then
  pushd "${SIMULATOR}"
  bash build.sh build_tool
  bazel build -c opt \
      //sim_control/dynamic_model/echo_lincoln_pipeline:echo_lincoln_bin
  popd
fi
