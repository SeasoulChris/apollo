#!/usr/bin/env bash

LOCAL_JOB="$1"

# Config.
CORES=4
ENV=py27

source /usr/local/miniconda2/bin/activate ${ENV}
source /apollo/scripts/apollo_base.sh

spark-submit --master "local[${CORES}]" "${LOCAL_JOB}"
