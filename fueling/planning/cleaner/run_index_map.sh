#!/usr/bin/env bash

bazel run //fueling/planning/cleaner:index_map_to_redis  -- --cloud --memory=10