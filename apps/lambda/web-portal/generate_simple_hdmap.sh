#!/usr/bin/env bash

INPUT_DATA_PATH=$1
ZONE_ID=$2
LIDAR_TYPE=$3
BASH_ARGS=$4
PY_ARGS=$5

# Go to apollo-fuel root.
cd /apollo/modules/data/fuel

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FUELING="/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip"
SUBMITTER="tools/submit-job-to-k8s.sh --partner_storage_writable --fueling ${FUELING} ${BASH_ARGS}"
JOB="fueling/map/generate_maps.py"
ENV="fuel-py36"
${SUBMITTER} --env ${ENV} --workers 5 --cpu 5 --memory 40g ${JOB} ${PY_ARGS} \
    --input_data_path="${INPUT_DATA_PATH}" --zone_id=$ZONE_ID --lidar_type="$LIDAR_TYPE"
