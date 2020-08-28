#!/usr/bin/env bash

bazel run //fueling/perception/pointpillars/pipelines:pointpillars_export_onnx_pipeline  -- \
--cloud \
--gpu=1 \
--config_path=/mnt/bos/modules/perception/pointpillars/config/all.pp.mhead.config \
--model_path=/mnt/bos/modules/perception/pointpillars/models/voxelnet-58650.tckpt \
--save_onnx_path=/mnt/bos/modules/perception/pointpillars/onnx/
