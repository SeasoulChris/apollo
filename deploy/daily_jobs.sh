#!/usr/bin/env bash

# Jobs that run once everyday at 1:00 a.m.
# Crontab example: 0 1 * * * /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Job: Data jobs.
JOB="fueling/data/daily-data-jobs.py"
./tools/submit-job-to-k8s.sh --fueling ${FUELING} --workers 15 --memory 24g --disk 800 ${JOB}

# Job: Control profiling.
JOB="fueling/control/control_profiling/control-profiling-metrics.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g ${JOB}
JOB="fueling/control/control_profiling/control-profiling-visualization.py"
CONDA_ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g -e ${CONDA_ENV} ${JOB}
