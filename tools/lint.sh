#!/usr/bin/env bash
#
# Lint a file or a directory, or lint the fueling package by default.
# Usage:
#   tools/lint.sh [path]

set -e

# TODO(?): Avoid as many exceptions as possible.
# E402: Module level import not at top of file
# W503: line break before binary operator
IGNORES="E402,W503"
LINT="pycodestyle --max-line-length 100 --show-source --ignore=${IGNORES}"
FLAKE="pyflakes"

function FatalDuildifier() {
  result=$( buildifier -v $@ 2>&1 )
  if [ ! -z "${result}" ]; then
    echo "${result}"
    return 1
  fi
}

function LintDir() {
  find "$1" -type f -name '*.py' | \
      grep -v 'fueling/common/record/kinglong/cybertron' | \
      grep -v 'prediction/learning/datasets/apollo_pedestrian_dataset/data_for_learning_pb2.py' | \
      xargs ${LINT}

  find "$1" -type f -name '*.py' | \
      grep -v 'fueling/common/logging.py' | \
      grep -v 'fueling/common/record/kinglong/cybertron' | \
      grep -v 'fueling/perception/pointpillars/second/pytorch/models/voxelnet.py' | \
      grep -v 'fueling/perception/pointpillars/second/pytorch/builder/second_builder.py' | \
      grep -v 'fueling/perception/pointpillars/second/builder/dataset_builder.py' | \
      xargs ${FLAKE}

  FatalDuildifier $( find "$1" -type f -name 'BUILD' )
}

PATH_ARG=$1
if [ -z "${PATH_ARG}" ]; then
  cd "$( dirname "${BASH_SOURCE[0]}" )/.."
  LintDir apps
  LintDir fueling
elif [ -d "${PATH_ARG}" ]; then
  LintDir "${PATH_ARG}"
elif [[ "${PATH_ARG}" = "*.py" ]]; then
  ${LINT} "${PATH_ARG}"
  ${FLAKE} "${PATH_ARG}"
elif [[ "${PATH_ARG}" = "*BUILD" ]]; then
  FatalDuildifier "${PATH_ARG}"
fi
