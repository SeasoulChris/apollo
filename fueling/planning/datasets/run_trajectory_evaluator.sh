#!/usr/bin/env bash

bazel run //fueling/planning/datasets:trajectory_evaluator -- --cloud --memory=50 --disk=100 --workers=8
