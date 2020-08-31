#!/usr/bin/env bash

bazel run //fueling/perception/pointpillars/pipelines:pointpillars_training_pipeline  -- \
--cloud \
--gpu=1 \
--config_path=/mnt/bos/modules/perception/pointpillars/config/all.pp.mhead.config \
--model_dir=/mnt/bos/modules/perception/pointpillars/models/ \
--pretrained_path=/mnt/bos/modules/perception/pointpillars/pretrained_model/voxelnet-58650.tckpt \
