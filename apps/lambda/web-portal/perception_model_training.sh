#!/usr/bin/env bash

INPUT_TRAINING_DATA_PATH=$1
OUTPUT_TRAINED_MODEL_PATH=$2
BASH_ARGS=$3
PY_ARGS=$4

# Go to apollo-fuel root.
cd /apollo/modules/data/fuel

set -e

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FUELING="/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip"
SUBMITER="./tools/submit-job-to-k8s.sh --fueling ${FUELING} ${BASH_ARGS}"

# Training.
JOB="./fueling/perception/YOLOv3/yolov3_training.py"
ENV="fuel-py36"
${SUBMITER} --env ${ENV} --workers 1 --memory 20g --gpu --partner_storage_writable ${JOB} ${PY_ARGS} \
    --input_training_data_path="${INPUT_TRAINING_DATA_PATH}" \
    --output_trained_model_path="${OUTPUT_TRAINED_MODEL_PATH}" 

