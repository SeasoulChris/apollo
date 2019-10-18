#!/usr/bin/env bash

INPUT_DATA_PATH=$1
ZONE_ID=$2
LIDAR_TYPE=$3
BASH_ARGS=$4
PY_ARGS=$5

# Go to apollo-fuel root.
cd /apollo/modules/data/fuel

set -e

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FUELING="/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip"
SUBMITER="tools/submit-job-to-k8s.sh --partner_storage_writable --fueling ${FUELING} ${BASH_ARGS}"

# Job: map generate base_map.txt
JOB="fueling/map/generate_base_map.py"
ENV="fuel-py27-cyber"
${SUBMITER} --env ${ENV} --workers 5 --cpu 5 --memory 40g ${JOB} ${PY_ARGS} \
    --input_data_path="${INPUT_DATA_PATH}"

# Job: map generate sim_map.txt or routing_map.txt
JOB="fueling/map/generate_sim_routing_map.py"
ENV="fuel-py27-cyber"
${SUBMITER} --env ${ENV} --workers 5 --cpu 5 --memory 60g ${JOB} ${PY_ARGS} \
    --input_data_path="${INPUT_DATA_PATH}"

# Job: map generate local_map
JOB="fueling/map/generate_local_map.py"
ENV="fuel-py27-cyber"
${SUBMITER} --env ${ENV} --workers 5 --cpu 5 --memory 60g ${JOB} ${PY_ARGS} \
    --input_data_path="${INPUT_DATA_PATH}" --ZONE_ID=$ZONE_ID --LIDAR_TYPE="$LIDAR_TYPE"
