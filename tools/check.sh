#!/usr/bin/env bash
# CI check.

CONTAINER="fuel"

cd $( dirname "${BASH_SOURCE[0]}" )

# Fail on any error.
set -e

RUNNER=""
if [ -z "$(which docker)" ]; then
  echo "Running inside container..."
else
  echo "Start container..."
  nohup bash login_container.sh > /dev/null 2>&1 &

  RUNNER="docker exec -it -u ${USER} ${CONTAINER}"
fi


echo "######################### Build #########################"
${RUNNER} bash /fuel/tools/build_local.sh

echo "######################### Test #########################"
# TODO(?): Contact the owner, fix the exceptions and enable all tests: "bazel test //..."
${RUNNER} bazel test $(bazel query //... \
    except //fueling/planning/cleaner:data_cleaner_test \
    except //fueling/profiling/control:multi_job_control_profiling_metrics_test
)

echo "######################### Lint #########################"
${RUNNER} bash /fuel/tools/lint.sh /fuel/apps
${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/common
${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/data
${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/demo
${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/streaming

# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/control
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/learning
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/map
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/perception
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/planning
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/prediction
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/profiling
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling/simulation
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/learning_algorithms
# TODO(?): ${RUNNER} bash /fuel/tools/lint.sh /fuel/fueling  # Finally

echo "######################### All check passed! #########################"
