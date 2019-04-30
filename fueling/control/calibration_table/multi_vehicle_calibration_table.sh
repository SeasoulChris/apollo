#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

# # Make sure you are calling run_prod() instead of run_test()!
# # Feature extraction.
# JOB="fueling/control/calibration_table/multi-vehicle-feature-extraction.py"
# ENV="fuel-py27-cyber"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 16 --cpu 2 --memory 20g ${JOB}

# Data distribution visualization
JOB="fueling/control/calibration_table/multi-vehicle-data-visualization.py"
ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --env ${ENV} --workers 9 --cpu 2 --memory 20g ${JOB}

# # Training.
# JOB="fueling/control/calibration_table/multi-vehicle-calibration-table-training.py"
# ENV="fuel-py27"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 10 --cpu 20 --memory 200g ${JOB}

# Result visualization
# JOB="fueling/control/calibration_table/multi-vehicle-calibration-result-visualization.py"
# ENV="fuel-py27"
# ./tools/submit-job-to-k8s.sh --env ${ENV} --workers 12 --cpu 2 --memory 20g ${JOB}
