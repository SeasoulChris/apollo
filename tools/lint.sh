#!/usr/bin/env bash
#
# Lint a file or a directory, or lint the fueling package by default.
# Usage:
#   tools/lint.sh [path]

# TODO(?): Avoid as many exceptions as possible.
# E402: Module level import not at top of file
# W503: line break before binary operator
IGNORES="E402,W503"
CMD="pycodestyle --max-line-length 100 --show-source --ignore=${IGNORES}"

function LintDir {
  find "$1" -type f -name '*.py' | \
      grep -v '_pb2.py$' | \
      grep -v 'fueling/common/record/kinglong/cybertron' | \
      xargs ${CMD}
}

PATH_ARG=$1
if [ -z "${PATH_ARG}" ]; then
  LintDir /fuel/apps
  LintDir /fuel/fueling
  LintDir /fuel/learning_algorithms
elif [ -d "${PATH_ARG}" ]; then
  LintDir "${PATH_ARG}"
else
  ${CMD} "${PATH_ARG}"
fi
