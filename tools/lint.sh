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

function LintDir {
  find "$1" -type f -name '*.py' | \
      grep -v 'fueling/common/record/kinglong/cybertron' | \
      grep -v 'prediction/learning/datasets/apollo_pedestrian_dataset/data_for_learning_pb2.py' | \
      xargs ${LINT}

  # TODO(all): Cover all code.
  find "$1" -type f -name '*.py' | \
      grep -v 'apps/k8s/warehouse/display_util.py' | \
      grep -v 'fueling/common/logging.py' | \
      grep -v 'fueling/common/record/kinglong/cybertron' | \
      grep -v 'fueling/learning/network_utils.py' | \
      grep -e 'apps/' \
          -e 'fueling/com' \
          -e 'fueling/d' \
          -e 'fueling/l' \
          -e 'fueling/m' \
          -e 'fueling/s' \
          -e 'perception/sen' \
          -e 'perception/pi' \
          -e 'perception/pointpillars/t' \
          -e 'prediction/c' \
          | xargs ${FLAKE}
}

PATH_ARG=$1
if [ -z "${PATH_ARG}" ]; then
  LintDir /fuel/apps
  LintDir /fuel/fueling
elif [ -d "${PATH_ARG}" ]; then
  LintDir "${PATH_ARG}"
else
  ${LINT} "${PATH_ARG}"
  ${FLAKE} "${PATH_ARG}"
fi
