#!/usr/bin/env bash

LOCAL_JOB="$1"

# Config.
CORES=8

# Specify the version you use.
# source /usr/local/miniconda2/bin/activate py27
source /usr/local/miniconda2/bin/activate py37

spark-submit --master "local[${CORES}]" ${LOCAL_JOB}
