#!/usr/bin/env bash

bazel run //fueling/perception/pointpillars/pipelines:pointpillars_end_to_end_pipeline  -- \
--input_data_path=/data/perception_data/kitti \
--output_data_path=/fuel/fueling/perception/pointpillars/second/output_result


