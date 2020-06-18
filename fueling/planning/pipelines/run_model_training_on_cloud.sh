#!/usr/bin/env bash

bazel run //fueling/planning/pipelines:model_training_on_cloud  -- --cloud --memory=40 --workers=1 --gpu=1
