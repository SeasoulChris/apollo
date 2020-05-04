#!/usr/bin/env bash
#
# Lint a file or a directory, or lint the fueling package by default.
# Usage:
#   tools/lint.sh [path]

CMD="pycodestyle --max-line-length 100"

function LintDir {
  find "$1" -type f -name '*.py' | grep -v '_pb2.py$' | xargs ${CMD} '{}'
}

PATH_ARG=$1
if [ -z "${PATH_ARG}" ]; then
  LintDir /fuel/apps
  LintDir /fuel/fueling
  # LintDir /fuel/learning_algorithms
elif [ -d "${PATH_ARG}" ]; then
  LintDir "${PATH_ARG}"
else
  ${CMD} "${PATH_ARG}"
fi
