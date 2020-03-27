#!/usr/bin/env bash

bazel run //fueling/planning/cleaner:data_cleaner  -- --cloud --memory=50 --disk=300 --workers=3