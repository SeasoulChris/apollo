#!/usr/bin/env bash

bazel run //fueling/perception/pointpillars/pipelines:create_data_pipeline  -- \
--data_path=/data/perception_data/nuscenes_data/mini/test

bazel run //fueling/perception/pointpillars/pipelines:pointpillars_training_pipeline  -- \
--config_path=/fuel/fueling/perception/pointpillars/second/configs/nuscenes/all.pp.mhead.config \
--model_dir=/fuel/fueling/perception/pointpillars/second/models/ \
--pretrained_path=/fuel/fueling/perception/pointpillars/second/pretrained_model/voxelnet-58650.tckpt
