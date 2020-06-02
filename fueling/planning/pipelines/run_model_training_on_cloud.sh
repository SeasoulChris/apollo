#!/usr/bin/env bash

bazel run //fueling/planning/pipelines:model_training_on_cloud  -- --cloud --driver_memory=20 --memory=20 --workers=1 --gpu=1
