#!/usr/bin/env bash

BASH_ARGS=$1
PY_ARGS=$2
INPUT_DATA_PATH=$3

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FUELING="/mnt/bos/modules/data/jobs/deploy/20190531-133701_fueling.zip"
SUBMITER="/apollo/modules/data/fuel/tools/submit-job-to-k8s.sh --fueling ${FUELING} ${BASH_ARGS}"

# 1. Feature extraction.
JOB="/apollo/modules/data/fuel/fueling/control/calibration_table/multi-vehicle-feature-extraction.py"
ENV="fuel-py27-cyber"
${SUBMITER} --env ${ENV} --workers 9 --cpu 2 --memory 20g ${JOB} \
    ${PY_ARGS} --input_data_path="${INPUT_DATA_PATH}"

# TODO
# 2. Training
# 3. Result visualization
# 4. Data distribution visualization
# 5. Notify partner
