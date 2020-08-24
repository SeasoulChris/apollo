#!/usr/bin/env bash

bazel run //fueling/planning/converter:data_converter  -- --cloud --memory=10 --disk=100 --workers=10
