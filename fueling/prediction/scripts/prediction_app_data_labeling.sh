#!/usr/bin/env bash

set -e

# Preapre: Goto fuel root, checkout latest code.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.."

git remote update
git reset --hard origin/master

# Job: Prediction generate feature.X.bin
JOB="fueling/prediction/dump_feature_proto.py"
./tools/submit-job-to-k8s.py --worker_count=16 --worker_memory=10 --entrypoint=${JOB}

# Job: Prediction generate-labels for feature.X.bin
JOB="fueling/prediction/generate_labels.py"
./tools/submit-job-to-k8s.py --worker_count=16 --worker_memory=10 --entrypoint=${JOB}

# Job: Prediction merge labels
JOB="fueling/prediction/merge_labels.py"
./tools/submit-job-to-k8s.py --worker_count=16 --worker_memory=10 --entrypoint=${JOB}
