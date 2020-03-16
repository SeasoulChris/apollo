#!/usr/bin/env bash

bazel run //fueling/planning/cleaner:data_cleaner  -- --cloud --memory=80 --disk=80