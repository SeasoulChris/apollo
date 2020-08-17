#!/usr/bin/env bash

bazel run //fueling/planning/training_pipelines:trajectory_imitation_model_training_pipeline  -- \
--gpu_idx=0 \
--model_type=cnn_lstm_aux \
--train_set_dir== \
--validation_set_dir= \
--update_base_map=False \
--regions_list=sunnyvale_with_two_offices,san_mateo \
--model_save_dir=