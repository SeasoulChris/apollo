#!/usr/bin/env bash
# CI check.

CONTAINER="fuel"

cd $( dirname "${BASH_SOURCE[0]}" )

# Fail on any error.
set -e

echo "######################### Build #########################"
bash /fuel/tools/build_local.sh

echo "######################### Test #########################"
bazel test //...

echo "######################### Lint #########################"
bash /fuel/tools/lint.sh /fuel/apps
bash /fuel/tools/lint.sh /fuel/fueling/common
bash /fuel/tools/lint.sh /fuel/fueling/control
bash /fuel/tools/lint.sh /fuel/fueling/data
bash /fuel/tools/lint.sh /fuel/fueling/demo
bash /fuel/tools/lint.sh /fuel/fueling/learning
bash /fuel/tools/lint.sh /fuel/fueling/perception
bash /fuel/tools/lint.sh /fuel/fueling/planning
bash /fuel/tools/lint.sh /fuel/fueling/streaming

# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/map
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/prediction
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/profiling
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/simulation
# TODO(?): bash /fuel/tools/lint.sh /fuel/learning_algorithms
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling  # Finally

echo "######################### All check passed! #########################"
