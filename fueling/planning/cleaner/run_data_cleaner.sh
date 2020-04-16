#!/usr/bin/env bash

bazel run //fueling/planning/cleaner:data_cleaner  -- --cloud --memory=10 --disk=300 --workers=30