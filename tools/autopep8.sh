#!/usr/bin/env bash
#
# Fix a file or a directory, or fix the fueling package by default.
# Usage:
#   tools/autopep8.sh [path]

function FixDir {
  find "$1" -type f -name '*.py' -exec \
      autopep8 --in-place --max-line-length 100 '{}' \;
}

PATH_ARG=$1
if [ -z "${PATH_ARG}" ]; then
  cd "$( dirname "${BASH_SOURCE[0]}" )/.."
  FixDir fueling
elif [ -d "${PATH_ARG}" ]; then
  FixDir "${PATH_ARG}"
else
  autopep8 --in-place --max-line-length 100 "${PATH_ARG}"
fi
