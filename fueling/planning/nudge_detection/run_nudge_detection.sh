#!/usr/bin/env bash

bazel run //fueling/planning/nudge_detection:nudge_detection  -- --cloud --memory=20 --disk=100 --workers=1
