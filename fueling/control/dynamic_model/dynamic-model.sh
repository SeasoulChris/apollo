#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
JOB="fueling/control/feature_extraction/sample_set_feature_extraction.py"
ENV="fuel-py27-cyber"
./tools/submit-job-to-k8s.py --main=${JOB} --workers=16 --cpu=2 --memory=20 --env=${ENV}

# Training.
JOB="fueling/control/dynamic_model/dynamic_model_training.py"
ENV="fuel-py36"
./tools/submit-job-to-k8s.py --main=${JOB} --workers=2 --cpu=20 --memory=100 --env=${ENV}

# Model Evaluation
JOB="fueling/control/dynamic_model/dynamic_model_evaluation.py"
ENV="fuel-py36"
./tools/submit-job-to-k8s.py --main=${JOB} --workers=16 --cpu=2 --memory=20 --env=${ENV}

# Training Data Visualization
JOB="fueling/control/dynamic_model/dynamic_model_data_visualization.py"
ENV="fuel-py36"
./tools/submit-job-to-k8s.py --main=${JOB} --workers=2 --cpu=20 --memory=100 --env=${ENV}
