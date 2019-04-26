#!/usr/bin/env bash

set -e

# Preapre: Goto fuel root, checkout latest code.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.."

git remote update
git reset --hard origin/master

# Job: Prediction generate feature.X.bin
JOB="fueling/prediction/dump-feature-proto.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 10g ${JOB}

# Job: Prediction generate-labels for feature.X.bin
JOB="fueling/prediction/generate-labels.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 10g ${JOB}

# Job: Prediction merge labels
JOB="fueling/prediction/merge-labels.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 10g ${JOB}
