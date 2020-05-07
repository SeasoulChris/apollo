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
${RUNNER} bash /fuel/tools/build.sh

echo "######################### Test #########################"
${RUNNER} bazel test //...

echo "######################### Lint #########################"
${RUNNER} bash /fuel/tools/lint.sh
