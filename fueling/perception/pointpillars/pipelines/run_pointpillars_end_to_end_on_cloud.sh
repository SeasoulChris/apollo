#!/usr/bin/env bash

DATA_PATH=$1
CONFIG_PATH="/mnt/bos/modules/perception/pointpillars/config/all.pp.mhead.config"
MODEL_DIR="/mnt/bos/modules/perception/pointpillars/models/"
PRETRAINED_PATH="/mnt/bos/modules/perception/pointpillars/pretrained_model/voxelnet-58650.tckpt"

bazel run //fueling/perception/pointpillars/pipelines:pointpillars_end_to_end_pipeline  -- \
--cloud \
--gpu=1 \
--data_path=$DATA_PATH \
--config_path=$CONFIG_PATH \
--model_dir=$MODEL_DIR \
--pretrained_path=$PRETRAINED_PATH
