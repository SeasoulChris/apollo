#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
JOB="fueling/control/feature_extraction/sample-set-feature-extraction.py"
ENV="fuel-py27-cyber"
./tools/submit-job-to-k8s.sh --job "${JOB}" --env "${ENV}"

# Training and evaluation
JOB="fueling/control/dynamic-model.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --job "${JOB}" --env "${ENV}"
