#!/usr/bin/env bash

BASH_ARGS=$1
PY_ARGS=$2

# Go to apollo-fuel root.
cd /apollo/modules/data/fuel

set -e

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FUELING="/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip"
SUBMITER="./tools/submit-job-to-k8s.sh --env fuel-py36 --fueling ${FUELING} ${BASH_ARGS}"
JOB="fueling/control/calibration_table/vehicle_calibration.py"
${SUBMITER} --workers 6 --cpu 4 --memory 60g ${JOB} ${PY_ARGS}
