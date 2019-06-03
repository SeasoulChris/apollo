#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
JOB="fueling/control/feature_extraction/sample-set-feature-extraction.py"
ENV="fuel-py27-cyber"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 16 --cpu 2 --memory 20g ${JOB}

# Training.
JOB="fueling/control/dynamic_model/dynamic-model-training.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 2 --cpu 20 --memory 100g ${JOB}

# Model Evaluation
JOB="fueling/control/dynamic_model/dynamic-model-evaluation.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 16 --cpu 2 --memory 20g ${JOB}

# Training Data Visualization
JOB="fueling/control/dynamic_model/dynamic-model-data-visualization.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 2 --cpu 20 --memory 100g ${JOB}
