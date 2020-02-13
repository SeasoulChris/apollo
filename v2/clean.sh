#!/usr/bin/env bash

find apps fueling learning_algorithms -name BUILD.v2 | \
while read filename; do
  rm -f "${filename%.*}"
done
