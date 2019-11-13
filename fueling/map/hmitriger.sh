#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

set -e

SUBMITTER="./tools/submit-job-to-k8s.py --workers=5 --cpu=5 --memory=60 --wait"
JOB_ID=$(date +%Y-%m-%d-%H)
INPUT_DATA_PATH="test/simplehdmap"
ZONE_ID=50
LIDAR_TYPE="velodyne16"

# Job: map generate base_map.txt
JOB="fueling/map/generate_base_map.py"
ENV="fuel-py27-cyber"
FLAGS="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH}"
${SUBMITTER} --main=${JOB} --env=${ENV} --flags="${FLAGS}"

# Job: map generate for sim_map.txt or routing_map.txt
JOB="fueling/map/generate_sim_routing_map.py"
ENV="fuel-py27-cyber"
FLAGS="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH}"
${SUBMITTER} --main=${JOB} --env=${ENV} --flags="${FLAGS}"

# Job: map generate local_map
JOB="fueling/map/generate_local_map.py"
ENV="fuel-py27-cyber"
FLAGS="--job_id=${JOB_ID} --input_data_path=${INPUT_DATA_PATH} \
       --zone_id=$ZONE_ID --lidar_type=$LIDAR_TYPE"
${SUBMITTER} --main=${JOB} --env=${ENV} --flags="${FLAGS}"
