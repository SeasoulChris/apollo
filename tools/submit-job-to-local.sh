#!/usr/bin/env bash

LOCAL_JOB="$1"

# Config.
CORES=4
ENV=py27

CONDA=/usr/local/miniconda2/bin/conda
# Update your conda and env:
#   ${CONDA} update -n base -c defaults conda
   ${CONDA} env update -f /apollo/modules/data/fuel/cluster/${ENV}-conda.yaml

source ${CONDA} activate ${ENV}
source /apollo/scripts/apollo_base.sh

spark-submit --master "local[${CORES}]" "${LOCAL_JOB}"
