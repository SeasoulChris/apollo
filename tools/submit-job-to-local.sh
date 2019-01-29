#!/usr/bin/env bash
#
# Prerequisite:
#   sudo /usr/local/miniconda2/bin/conda env update -f \
#        /apollo/modules/data/fuel/cluster/py27-conda.yaml
#   sudo /usr/local/miniconda2/bin/conda env update -f \
#        /apollo/modules/data/fuel/cluster/py37-conda.yaml

LOCAL_JOB="$1"

# Update config.
CORES=8

# Specify the version you use.
# source /usr/local/miniconda2/bin/activate py27
source /usr/local/miniconda2/bin/activate py37

spark-submit --master "local[${CORES}]" ${LOCAL_JOB}
