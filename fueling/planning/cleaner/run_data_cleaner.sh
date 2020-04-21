#!/usr/bin/env bash

bazel run //fueling/planning/cleaner:data_cleaner  -- --cloud --memory=5 --disk=100 --workers=40