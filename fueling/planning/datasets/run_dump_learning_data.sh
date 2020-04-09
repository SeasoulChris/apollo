#!/usr/bin/env bash

bazel run //fueling/planning/datasets:dump_learning_data  -- --cloud --memory=80 --disk=80
