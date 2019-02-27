#!/bin/bash
set -e

/apollo-simulator/bazel-bin/logsim_generator/logsim_generator_executable --input_dir=$1 --output_dir=$2 --scenario_map_dir=$3 --alsologtostderr
