#!/usr/bin/env bash

bazel run //fueling/planning/routing_generator:routing_generator  -- --cloud --memory=10 --disk=100 --workers=1
