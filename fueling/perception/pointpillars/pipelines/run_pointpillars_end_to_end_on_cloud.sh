#!/usr/bin/env bash

INPUT_DATA_PATH=$1
OUTPUT_DATA_PATH=$2

bazel run //fueling/perception/pointpillars/pipelines:pointpillars_end_to_end_pipeline  -- \
--cloud \
--gpu=1 \
--input_data_path=$INPUT_DATA_PATH \
--output_data_path=$OUTPUT_DATA_PATH
