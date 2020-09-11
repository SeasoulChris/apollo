#!/usr/bin/env bash

bazel test //fueling/perception/pointpillars/pipelines:create_kitti_data_pipeline_test
bazel test //fueling/perception/pointpillars/pipelines:pointpillars_training_pipeline_test
bazel test //fueling/perception/pointpillars/pipelines:pointpillars_export_onnx_pipeline_test
