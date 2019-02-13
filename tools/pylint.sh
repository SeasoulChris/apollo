#!/usr/bin/env bash
#
# Check a file or a directory, or check the fueling package by default.
# Usage:
#   tools/pylint.sh [path]

function LintDir {
  find "$1" -type f | grep '\.py$' | grep -v '__init__\.py' | \
      xargs -L 1 pylint
}

PATH_ARG=$1
if [ -z "${PATH_ARG}" ]; then
  cd "$( dirname "${BASH_SOURCE[0]}" )/.."
  LintDir fueling
elif [ -d "${PATH_ARG}" ]; then
  LintDir "${PATH_ARG}"
else
  pylint "${PATH_ARG}"
fi
