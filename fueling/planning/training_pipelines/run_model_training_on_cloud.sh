#!/usr/bin/env bash

bazel run //fueling/planning/training_pipelines:model_training_on_cloud  -- \
--cloud \
--memory=40 \
--workers=1 \
--gpu=1 \
--model_type=cnn \
--train_set_dir=/mnt/bos/modules/planning/imitation/training_data/ \
--validation_set_dir=/mnt/bos/modules/planning/imitation/validation_data/ \
--renderer_config_file=/mnt/bos/modules/planning/imitation/semantic_map_features/planning_semantic_map_config.pb.txt \
--renderer_base_map_img_dir=/mnt/bos/modules/planning/imitation/semantic_map_features \
--renderer_base_map_data_dir=/mnt/bos/code/baidu/adu-lab/apollo-map/ \
--model_save_dir=/mnt/bos/modules/planning/imitation/model/ \
--gpu_idx=0 \
--update_base_map=False \
--regions_list=sunnyvale_with_two_offices,san_mateo \
