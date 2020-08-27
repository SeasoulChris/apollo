#!/usr/bin/env bash

bazel run //fueling/perception/pointpillars/pipelines:create_data_pipeline  -- \
--cloud \
--data_path=/mnt/bos/modules/perception/pointpillars/data/ \

