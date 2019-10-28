#!/usr/bin/env bash
#
# Fix a file or a directory, or fix the fueling package by default.
# Usage:
#   tools/autopep8.sh [path]

CMD="autopep8 --in-place --max-line-length 100 -a"

if [ -z $(which autopep8) ]; then
  sudo pip install autopep8
fi

function FixDir {
  find "$1" -type f -name '*.py' -exec ${CMD} '{}' \;
}

PATH_ARG=$1
if [ -z "${PATH_ARG}" ]; then
  cd "$( dirname "${BASH_SOURCE[0]}" )/.."
  FixDir fueling
elif [ -d "${PATH_ARG}" ]; then
  FixDir "${PATH_ARG}"
else
  ${CMD} "${PATH_ARG}"
fi
