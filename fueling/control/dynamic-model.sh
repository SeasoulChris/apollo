#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
JOB="fueling/control/feature_extraction/sample-set-feature-extraction.py"
ENV="fuel-py27-cyber"
./tools/submit-job-to-k8s.sh --job "${JOB}" --env "${ENV}" \
    --workers 15 --worker_cpu 2 --worker_memory 24g

# Training.
# TODO(jiaxuan): Change the job name accordingly.
JOB="fueling/control/dynamic-model-training.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --job "${JOB}" --env "${ENV}" \
    --workers 1 --worker_cpu 24 --worker_memory 200g

# Evaluation
JOB="fueling/control/dynamic-model-evaluation.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --job "${JOB}" --env "${ENV}" \
    --workers 15 --worker_cpu 2 --worker_memory 24g
