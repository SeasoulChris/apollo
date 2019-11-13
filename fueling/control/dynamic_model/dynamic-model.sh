#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

SUBMITTER="./tools/submit-job-to-k8s.py --wait"
# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
JOB="fueling/control/feature_extraction/sample_set_feature_extraction.py"
${SUBMITTER} --main=${JOB} --workers=16 --cpu=2 --memory=20

# Training.
JOB="fueling/control/dynamic_model/dynamic_model_training.py"
${SUBMITTER} --main=${JOB} --workers=2 --cpu=20 --memory=100

# Model Evaluation
JOB="fueling/control/dynamic_model/dynamic_model_evaluation.py"
${SUBMITTER} --main=${JOB} --workers=16 --cpu=2 --memory=20

# Training Data Visualization
JOB="fueling/control/dynamic_model/dynamic_model_data_visualization.py"
${SUBMITTER} --main=${JOB} --workers=2 --cpu=20 --memory=100
