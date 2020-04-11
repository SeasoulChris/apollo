#!/usr/bin/env bash

bazel run //fueling/planning/datasets:learning_data_generator  -- --cloud --memory=80 --disk=80
