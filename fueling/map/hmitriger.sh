#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

set -e

SUBMITTER="./tools/submit-job-to-k8s.py --worker_count=5 --worker_cpu=5 --worker_memory=60"
JOB_ID=$(date +%Y-%m-%d-%H)
INPUT_DATA_PATH="test/simplehdmap"
ZONE_ID=50
LIDAR_TYPE="velodyne16"

# Job: map generate base_map.txt
JOB="fueling/map/generate_base_map.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --conda_env=${ENV} --entrypoint=${JOB} \
    --job_flags="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH}"

# Job: map generate for sim_map.txt or routing_map.txt
JOB="fueling/map/generate_sim_routing_map.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --conda_env=${ENV} --entrypoint=${JOB} \
    --job_flags="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH}"

# Job: map generate local_map
JOB="fueling/map/generate_local_map.py"
ENV="fuel-py27-cyber"
${SUBMITTER} --conda_env=${ENV} --entrypoint=${JOB} \
    --job_flags="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH} \
                 --zone_id=$ZONE_ID --lidar_type=$LIDAR_TYPE"
