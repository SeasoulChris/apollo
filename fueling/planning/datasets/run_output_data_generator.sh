#!/usr/bin/env bash

bazel run //fueling/planning/datasets:output_data_generator  -- --cloud --memory=50 --disk=100 --workers=8 
