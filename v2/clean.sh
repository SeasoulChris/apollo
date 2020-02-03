#!/usr/bin/env bash

find fueling deps -name BUILD.v2 | \
while read filename; do
  rm -f "${filename%.*}"
done
