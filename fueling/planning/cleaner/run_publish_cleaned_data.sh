#!/usr/bin/env bash

bazel run //fueling/planning/cleaner:publish_cleaned_data  -- --cloud --memory=80 --disk=80