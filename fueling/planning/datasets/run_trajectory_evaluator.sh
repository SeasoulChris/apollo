#!/usr/bin/env bash

bazel run //fueling/planning/datasets:trajectory_evaluator  -- --cloud --memory=80 --disk=80
