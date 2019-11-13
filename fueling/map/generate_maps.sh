#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

SUBMITTER="./tools/submit-job-to-k8s.py --workers=5 --cpu=5 --memory=60"
INPUT_DATA_PATH="test/simplehdmap"
ZONE_ID=50
LIDAR_TYPE="velodyne16"

JOB="fueling/map/generate_maps.py"
FLAGS="--input_data_path=${INPUT_DATA_PATH} --zone_id=$ZONE_ID --lidar_type=$LIDAR_TYPE"
${SUBMITTER} --main=${JOB} --flags="${FLAGS}"
