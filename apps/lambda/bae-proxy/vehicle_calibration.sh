#!/usr/bin/env bash

BASH_ARGS=$1
PY_ARGS=$2
INPUT_DATA_PATH=$3

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /apollo/modules/data/fuel

FUELING="/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip"
SUBMITER="./tools/submit-job-to-k8s.sh --fueling ${FUELING} ${BASH_ARGS}"

# Feature extraction.
JOB="./fueling/control/calibration_table/multi-job-feature-extraction.py"
ENV="fuel-py27-cyber"
${SUBMITER} --env ${ENV} --workers 10 --cpu 1 --memory 40g ${JOB} ${PY_ARGS} \
    --input_data_path="${INPUT_DATA_PATH}"

# Training.
JOB="./fueling/control/calibration_table/multi-job-train.py"
ENV="fuel-py27"
${SUBMITER} --env ${ENV} --workers 5 --cpu 5 --memory 60g ${JOB} ${PY_ARGS}

# Result visualization
JOB="./fueling/control/calibration_table/multi-job-result-visualization.py"
ENV="fuel-py27"
${SUBMITER} --env ${ENV} --workers 10 --cpu 1 --memory 20g ${JOB} ${PY_ARGS}

# Data distribution visualization
JOB="./fueling/control/calibration_table/multi-job-data-distribution.py"
ENV="fuel-py27"
${SUBMITER} --env ${ENV} --workers 10 --cpu 1 --memory 20g ${JOB} ${PY_ARGS}
