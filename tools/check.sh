#!/usr/bin/env bash
# CI check.

CONTAINER="fuel"

cd $( dirname "${BASH_SOURCE[0]}" )

# Fail on any error.
set -e

echo "######################### Build #########################"
bash /fuel/tools/build_local.sh

echo "######################### Test #########################"
# TODO(?): Contact the owner, fix the exceptions and enable all tests: "bazel test //..."
bazel test $(bazel query //... \
    except //fueling/profiling/control:multi_job_control_profiling_metrics_test
)

echo "######################### Lint #########################"
bash /fuel/tools/lint.sh /fuel/apps
bash /fuel/tools/lint.sh /fuel/fueling/common
bash /fuel/tools/lint.sh /fuel/fueling/data
bash /fuel/tools/lint.sh /fuel/fueling/demo
bash /fuel/tools/lint.sh /fuel/fueling/perception
bash /fuel/tools/lint.sh /fuel/fueling/planning
bash /fuel/tools/lint.sh /fuel/fueling/streaming
bash /fuel/tools/lint.sh /fuel/fueling/control

# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/learning
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/map
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/prediction
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/profiling
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling/simulation
# TODO(?): bash /fuel/tools/lint.sh /fuel/learning_algorithms
# TODO(?): bash /fuel/tools/lint.sh /fuel/fueling  # Finally

echo "######################### All check passed! #########################"
