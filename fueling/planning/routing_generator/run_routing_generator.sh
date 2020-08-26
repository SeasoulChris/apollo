#!/usr/bin/env bash

bazel run //fueling/planning/routing_generator:routing_generator  -- --cloud --driver_memory=10 --memory=20 --disk=100 --workers=10
