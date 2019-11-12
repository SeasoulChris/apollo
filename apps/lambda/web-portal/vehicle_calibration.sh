#!/usr/bin/env bash

INPUT_DATA_PATH=$1
BASH_ARGS=$2
PY_ARGS=$3

# Go to apollo-fuel root.
cd /apollo/modules/data/fuel

set -e

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FUELING="/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip"
SUBMITER="./tools/submit-job-to-k8s.sh --fueling ${FUELING} ${BASH_ARGS}"

# Feature extraction.
JOB="./fueling/control/calibration_table/multi_job_feature_extraction.py"
ENV="fuel-py27-cyber"
${SUBMITER} --env ${ENV} --workers 10 --cpu 1 --memory 40g ${JOB} ${PY_ARGS} \
    --input_data_path="${INPUT_DATA_PATH}"

# Training, result visualization and distribution.
JOB="./fueling/control/calibration_table/multi_job_train_vis_dist.py"
ENV="fuel-py36"
${SUBMITER} --env ${ENV} --workers 5 --cpu 5 --memory 60g ${JOB} ${PY_ARGS}
